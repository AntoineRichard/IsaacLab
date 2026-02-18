# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import convert_quat
from isaaclab.utils.warp.kernels import (
    add_forces_to_dual_buffers,
    add_raw_wrench_buffers,
    compose_wrench_to_body_frame,
    set_forces_to_dual_buffers,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection


class WrenchComposer:
    def __init__(self, asset: Articulation | RigidObject | RigidObjectCollection) -> None:
        """Wrench composer with dual-buffer architecture (global + local).

        Forces and torques are stored in two separate pairs of buffers:
        - **Global buffers** (``_global_force_w``, ``_global_torque_w``): world-frame forces/torques.
        - **Local buffers** (``_local_force_b``, ``_local_torque_b``): body-frame forces/torques.

        At apply time, :meth:`compose_to_body_frame` rotates global forces into the body frame
        using the current body quaternion and sums with local forces, producing a single
        body-frame wrench that can be applied with ``is_global=False``.

        Args:
            asset: Asset to use.
        """
        self.num_envs = asset.num_instances
        # Avoid isinstance to prevent circular import issues, use attribute presence instead.
        if hasattr(asset, "num_bodies"):
            self.num_bodies = asset.num_bodies
        else:
            self.num_bodies = asset.num_objects
        self.device = asset.device
        self._asset = asset
        self._active = False
        self._dirty = False

        # Avoid isinstance here due to potential circular import issues; check by attribute presence instead.
        if hasattr(self._asset.data, "body_link_pos_w") and hasattr(self._asset.data, "body_link_quat_w"):
            self._get_link_position_fn = lambda a=self._asset: a.data.body_link_pos_w[..., :3]
            self._get_link_quaternion_fn = lambda a=self._asset: a.data.body_link_quat_w[..., :4]
        elif hasattr(self._asset.data, "object_link_pos_w") and hasattr(self._asset.data, "object_link_quat_w"):
            self._get_link_position_fn = lambda a=self._asset: a.data.object_link_pos_w[..., :3]
            self._get_link_quaternion_fn = lambda a=self._asset: a.data.object_link_quat_w[..., :4]
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        shape = (self.num_envs, self.num_bodies)

        # Input buffers: global (world-frame) and local (body-frame)
        self._global_force_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._global_torque_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._global_force_at_com_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._local_force_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._local_torque_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)

        # Output buffers: composed body-frame wrench
        self._out_force_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._out_torque_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)

        # Index arrays
        self._ALL_ENV_INDICES_WP = wp.from_torch(
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device), dtype=wp.int32
        )
        self._ALL_BODY_INDICES_WP = wp.from_torch(
            torch.arange(self.num_bodies, dtype=torch.int32, device=self.device), dtype=wp.int32
        )

        # Pinned torch views of output buffers (for PhysX apply calls)
        self._out_force_b_torch = wp.to_torch(self._out_force_b)
        self._out_torque_b_torch = wp.to_torch(self._out_torque_b)

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def global_force_w(self) -> wp.array:
        """Global (world-frame) force buffer. Shape: (num_envs, num_bodies) vec3f."""
        return self._global_force_w

    @property
    def global_torque_w(self) -> wp.array:
        """Global (world-frame) torque buffer. Shape: (num_envs, num_bodies) vec3f."""
        return self._global_torque_w

    @property
    def global_force_at_com_w(self) -> wp.array:
        """Global force applied at CoM (no positional torque). Shape: (num_envs, num_bodies) vec3f."""
        return self._global_force_at_com_w

    @property
    def local_force_b(self) -> wp.array:
        """Local (body-frame) force buffer. Shape: (num_envs, num_bodies) vec3f."""
        return self._local_force_b

    @property
    def local_torque_b(self) -> wp.array:
        """Local (body-frame) torque buffer. Shape: (num_envs, num_bodies) vec3f."""
        return self._local_torque_b

    @property
    def out_force_b(self) -> wp.array:
        """Composed output force in body frame. Shape: (num_envs, num_bodies) vec3f.

        If the output is stale (buffers were modified since last :meth:`compose_to_body_frame`),
        this will automatically recompose and emit a performance warning.
        """
        self._ensure_composed()
        return self._out_force_b

    @property
    def out_torque_b(self) -> wp.array:
        """Composed output torque in body frame. Shape: (num_envs, num_bodies) vec3f.

        If the output is stale (buffers were modified since last :meth:`compose_to_body_frame`),
        this will automatically recompose and emit a performance warning.
        """
        self._ensure_composed()
        return self._out_torque_b

    @property
    def out_force_b_as_torch(self) -> torch.Tensor:
        """Composed output force in body frame as torch tensor. Shape: (num_envs, num_bodies, 3).

        If the output is stale (buffers were modified since last :meth:`compose_to_body_frame`),
        this will automatically recompose and emit a performance warning.
        """
        self._ensure_composed()
        return self._out_force_b_torch

    @property
    def out_torque_b_as_torch(self) -> torch.Tensor:
        """Composed output torque in body frame as torch tensor. Shape: (num_envs, num_bodies, 3).

        If the output is stale (buffers were modified since last :meth:`compose_to_body_frame`),
        this will automatically recompose and emit a performance warning.
        """
        self._ensure_composed()
        return self._out_torque_b_torch

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques to the appropriate global or local buffer.

        Routes to global buffers when ``is_global=True``, local buffers when ``is_global=False``.
        Position offsets contribute additional torque via cross product.

        When ``is_global=True``:

        - Forces **with** positions are stored in the global positional buffer. The torque
          ``cross(P, F)`` is stored about the world origin and corrected at compose time.
        - Forces **without** positions are applied at the body's center of mass (no positional torque).
        - Torques are stored directly in the global torque buffer.

        When ``is_global=False``:

        - Forces and torques are stored in the local (body-frame) buffers.
        - Positions are local offsets from the link frame contributing ``cross(pos, F)`` torque.

        Args:
            forces: Forces. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            torques: Torques. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            positions: Application points for forces. Shape: (len(env_ids), len(body_ids), 3).
                Defaults to None.
                When ``is_global=False``, positions are local offsets from the link frame.
                When ``is_global=True``, positions are world-frame coordinates. If None,
                forces are applied at the body's center of mass (no positional torque).
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether forces and torques are in global frame. Defaults to False.
        """
        env_ids, body_ids = self._resolve_indices(env_ids, body_ids)

        if forces is None and torques is None:
            return
        if isinstance(forces, torch.Tensor):
            forces = wp.from_torch(forces, dtype=wp.vec3f)
        if isinstance(torques, torch.Tensor):
            torques = wp.from_torch(torques, dtype=wp.vec3f)
        if isinstance(positions, torch.Tensor):
            positions = wp.from_torch(positions, dtype=wp.vec3f)

        self._active = True
        self._dirty = True

        wp.launch(
            add_forces_to_dual_buffers,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def set_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Set forces and torques, replacing all existing values in every buffer.

        All 5 input buffers are cleared first, then the provided values are written to the
        appropriate buffer. Use :meth:`add_forces_and_torques` to accumulate on top of existing values.

        Routes to global buffers when ``is_global=True``, local buffers when ``is_global=False``.
        Position offsets contribute additional torque via cross product.

        When ``is_global=True``:

        - Forces **with** positions are stored in the global positional buffer. The torque
          ``cross(P, F)`` is stored about the world origin and corrected at compose time.
        - Forces **without** positions are applied at the body's center of mass (no positional torque).
        - Torques are stored directly in the global torque buffer.

        When ``is_global=False``:

        - Forces and torques are stored in the local (body-frame) buffers.
        - Positions are local offsets from the link frame contributing ``cross(pos, F)`` torque.

        Args:
            forces: Forces. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            torques: Torques. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            positions: Application points for forces. Shape: (len(env_ids), len(body_ids), 3).
                Defaults to None.
                When ``is_global=False``, positions are local offsets from the link frame.
                When ``is_global=True``, positions are world-frame coordinates. If None,
                forces are applied at the body's center of mass (no positional torque).
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether forces and torques are in global frame. Defaults to False.
        """
        env_ids, body_ids = self._resolve_indices(env_ids, body_ids)

        if forces is None and torques is None:
            return
        if forces is None:
            forces = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        elif isinstance(forces, torch.Tensor):
            forces = wp.from_torch(forces, dtype=wp.vec3f)
        if torques is None:
            torques = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        elif isinstance(torques, torch.Tensor):
            torques = wp.from_torch(torques, dtype=wp.vec3f)
        if positions is None:
            positions = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        elif isinstance(positions, torch.Tensor):
            positions = wp.from_torch(positions, dtype=wp.vec3f)

        self._active = True
        self._dirty = True

        # Clear all input buffers first — set means "replace everything"
        self._global_force_w.zero_()
        self._global_torque_w.zero_()
        self._global_force_at_com_w.zero_()
        self._local_force_b.zero_()
        self._local_torque_b.zero_()

        wp.launch(
            set_forces_to_dual_buffers,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def add_raw_buffers_from(self, other: WrenchComposer):
        """Element-wise add another composer's 4 input buffers into this composer's buffers.

        Args:
            other: Source wrench composer whose buffers will be added.
        """
        self._dirty = True
        wp.launch(
            add_raw_wrench_buffers,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                other._global_force_w,
                other._global_torque_w,
                other._global_force_at_com_w,
                other._local_force_b,
                other._local_torque_b,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

    def compose_to_body_frame(self):
        """Compose global and local buffers into body-frame output.

        Fetches current link positions and quaternions. Global torques stored about
        the world origin are corrected to be about the current CoM via
        ``-cross(link_pos, global_force)``, then rotated into body frame via
        ``quat_rotate_inv`` and summed with local forces/torques. Result is written
        to :attr:`out_force_b` / :attr:`out_torque_b`.
        """
        link_quaternions = wp.from_torch(
            convert_quat(self._get_link_quaternion_fn().clone(), to="xyzw"), dtype=wp.quatf
        )
        link_positions = wp.from_torch(self._get_link_position_fn().clone(), dtype=wp.vec3f)

        wp.launch(
            compose_wrench_to_body_frame,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                link_positions,
                link_quaternions,
                self._out_force_b,
                self._out_torque_b,
            ],
            device=self.device,
        )
        self._dirty = False

    def reset(self, env_ids: wp.array | torch.Tensor | None = None, env_mask: wp.array | None = None):
        """Reset all input and output buffers to zero.

        Args:
            env_ids: Environment indices to reset. If None or slice(None), resets all.
            env_mask: Environment mask (unused, kept for API compatibility).

        Raises:
            ValueError: If env_ids is a slice other than slice(None).
        """
        if env_ids is None or env_ids == slice(None):
            self._global_force_w.zero_()
            self._global_torque_w.zero_()
            self._global_force_at_com_w.zero_()
            self._local_force_b.zero_()
            self._local_torque_b.zero_()
            self._out_force_b.zero_()
            self._out_torque_b.zero_()
            self._active = False
        else:
            if isinstance(env_ids, slice):
                raise ValueError(
                    f"WrenchComposer.reset() does not support arbitrary slices, got {env_ids!r}. "
                    "Use None, slice(None), or explicit index arrays instead."
                )
            indices = env_ids
            if isinstance(env_ids, torch.Tensor):
                indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            elif isinstance(env_ids, list):
                indices = wp.array(env_ids, dtype=wp.int32, device=self.device)

            self._global_force_w[indices].zero_()
            self._global_torque_w[indices].zero_()
            self._global_force_at_com_w[indices].zero_()
            self._local_force_b[indices].zero_()
            self._local_torque_b[indices].zero_()
            self._out_force_b[indices].zero_()
            self._out_torque_b[indices].zero_()
        self._dirty = False

    def _ensure_composed(self):
        """Ensure output buffers are up-to-date. If dirty, recomposes and warns."""
        if self._dirty:
            warnings.warn(
                "WrenchComposer: accessing output property triggered compose_to_body_frame() kernel launch. "
                "Call compose_to_body_frame() explicitly before accessing output properties to avoid this overhead. "
                "If you only need forces/torques in a single frame, use the raw buffer properties instead "
                "(global_force_w, global_torque_w, local_force_b, local_torque_b) which require no composition.",
                stacklevel=3,
            )
            self.compose_to_body_frame()

    def _resolve_indices(
        self,
        env_ids: wp.array | torch.Tensor | None,
        body_ids: wp.array | torch.Tensor | None,
    ) -> tuple[wp.array, wp.array]:
        """Resolve env and body indices to warp arrays."""
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES_WP
        elif isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        elif isinstance(env_ids, list):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, slice):
            if env_ids == slice(None):
                env_ids = self._ALL_ENV_INDICES_WP
            else:
                raise ValueError(f"Doesn't support slice input for env_ids: {env_ids}")

        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES_WP
        elif isinstance(body_ids, torch.Tensor):
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        elif isinstance(body_ids, list):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        elif isinstance(body_ids, slice):
            if body_ids == slice(None):
                body_ids = self._ALL_BODY_INDICES_WP
            else:
                raise ValueError(f"Doesn't support slice input for body_ids: {body_ids}")

        return env_ids, body_ids
