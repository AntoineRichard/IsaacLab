# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for the Newton actuator fast path."""

from collections.abc import Sequence
from typing import Protocol

import torch
import warp as wp

from isaaclab.actuators import ActuatorBase, ImplicitActuator


class _GroupLayoutLike(Protocol):
    """Host metadata required to build actuator ownership rows."""

    name: str
    actuator_type: type[ActuatorBase]
    joint_indices: Sequence[int]


# ---------------------------------------------------------------------------
# Adapter / per-actuator helper kernels: per-DOF zeroing, env-mask building,
# per-DOF env-mask projection (used by :meth:`NewtonActuatorAdapter.reset`),
# and a partial scatter for DR gain updates that overwrites only the cells
# in a (env_ids × joint_ids) sub-grid of a Newton ``Actuator``'s controller
# parameter array. Used on the PhysX backend (no Newton view available);
# the Newton backend uses ``ArticulationView.set_actuator_parameter`` instead.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def zero_at_indices_kernel(data: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.uint32)):
    """Zero a flat ``data`` buffer at the given flat ``indices``."""
    i = wp.tid()
    data[indices[i]] = 0.0


@wp.kernel(enable_backward=False)
def gather_canonical_range_to_controller(
    canonical: wp.array2d(dtype=wp.float32),
    canonical_slots: wp.array(dtype=wp.int32),
    controller_local_slots: wp.array(dtype=wp.int32),
    controller_stride: int,
    controller_values: wp.array(dtype=wp.float32),
):
    """Gather one canonical world-major range into fixed controller slots."""
    env_id, compact_dof = wp.tid()
    controller_slot = env_id * controller_stride + controller_local_slots[compact_dof]
    controller_values[controller_slot] = canonical[env_id, canonical_slots[compact_dof]]


@wp.kernel(enable_backward=False)
def scatter_controller_range_to_canonical(
    controller_values: wp.array(dtype=wp.float32),
    canonical_slots: wp.array(dtype=wp.int32),
    controller_local_slots: wp.array(dtype=wp.int32),
    controller_stride: int,
    canonical: wp.array2d(dtype=wp.float32),
):
    """Scatter fixed controller slots into one canonical world-major range."""
    env_id, compact_dof = wp.tid()
    controller_slot = env_id * controller_stride + controller_local_slots[compact_dof]
    canonical[env_id, canonical_slots[compact_dof]] = controller_values[controller_slot]


@wp.kernel(enable_backward=False)
def scatter_canonical_range_to_joint(
    canonical: wp.array2d(dtype=wp.float32),
    compact_joint_ids: wp.array(dtype=wp.int32),
    canonical_slots: wp.array(dtype=wp.int32),
    owner_slots: wp.array(dtype=wp.int32),
    joint: wp.array2d(dtype=wp.float32),
):
    """Publish a compact canonical exact-type range into articulation joint order."""
    env_id, compact_dof = wp.tid()
    joint_id = compact_joint_ids[compact_dof]
    canonical_slot = canonical_slots[compact_dof]
    if owner_slots[joint_id] == canonical_slot:
        joint[env_id, joint_id] = canonical[env_id, canonical_slot]


@wp.kernel(enable_backward=False)
def scatter_native_effort_to_command(
    native_effort: wp.array2d(dtype=wp.float32),
    native_owner: wp.array(dtype=wp.int32),
    command_effort: wp.array2d(dtype=wp.float32),
):
    """Publish only native-owned physical effort slots into the merged command."""
    env_id, joint_id = wp.tid()
    if native_owner[joint_id] != 0:
        command_effort[env_id, joint_id] = native_effort[env_id, joint_id]


@wp.kernel(enable_backward=False)
def restore_backend_effort_from_command(
    command_effort: wp.array2d(dtype=wp.float32),
    native_touched: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    sim_buffers_are_backend_order: bool,
    backend_effort: wp.array2d(dtype=wp.float32),
):
    """Restore non-winning native slots from the processed user-order command."""
    env_id, user_joint = wp.tid()
    if native_touched[user_joint] != 0:
        backend_joint = user_joint
        if sim_buffers_are_backend_order:
            backend_joint = user_to_backend[user_joint]
        backend_effort[env_id, backend_joint] = command_effort[env_id, user_joint]


@wp.kernel(enable_backward=False)
def scatter_canonical_range_to_backend(
    canonical: wp.array2d(dtype=wp.float32),
    compact_joint_ids: wp.array(dtype=wp.int32),
    canonical_slots: wp.array(dtype=wp.int32),
    owner_slots: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    sim_buffers_are_backend_order: bool,
    backend_effort: wp.array2d(dtype=wp.float32),
):
    """Write final native winners from compact canonical storage to backend effort."""
    env_id, compact_dof = wp.tid()
    user_joint = compact_joint_ids[compact_dof]
    canonical_slot = canonical_slots[compact_dof]
    if owner_slots[user_joint] == canonical_slot:
        backend_joint = user_joint
        if sim_buffers_are_backend_order:
            backend_joint = user_to_backend[user_joint]
        backend_effort[env_id, backend_joint] = canonical[env_id, canonical_slot]


@wp.kernel(enable_backward=False)
def merge_native_command_fields(
    raw_position: wp.array2d(dtype=wp.float32),
    raw_velocity: wp.array2d(dtype=wp.float32),
    raw_effort: wp.array2d(dtype=wp.float32),
    native_position_owner: wp.array(dtype=wp.int32),
    native_velocity_owner: wp.array(dtype=wp.int32),
    native_effort_owner: wp.array(dtype=wp.int32),
    processed_position: wp.array2d(dtype=wp.float32),
    processed_velocity: wp.array2d(dtype=wp.float32),
    processed_effort: wp.array2d(dtype=wp.float32),
):
    """Copy raw command fields only into native-owned public joint slots."""
    env_id, joint_id = wp.tid()
    if native_position_owner[joint_id] != 0:
        processed_position[env_id, joint_id] = raw_position[env_id, joint_id]
    if native_velocity_owner[joint_id] != 0:
        processed_velocity[env_id, joint_id] = raw_velocity[env_id, joint_id]
    if native_effort_owner[joint_id] != 0:
        processed_effort[env_id, joint_id] = raw_effort[env_id, joint_id]


@wp.kernel(enable_backward=False)
def set_mask_kernel(mask: wp.array(dtype=wp.bool), indices: wp.array(dtype=wp.int32)):
    """Set ``mask[indices[i]] = True`` for each ``i``. The mask must be pre-zeroed."""
    i = wp.tid()
    mask[indices[i]] = True


@wp.kernel(enable_backward=False)
def set_mask_int64_kernel(mask: wp.array(dtype=wp.bool), indices: wp.array(dtype=wp.int64)):
    """Set ``mask[indices[i]] = True`` for signed 64-bit indices."""
    i = wp.tid()
    mask[indices[i]] = True


@wp.kernel(enable_backward=False)
def set_mask_slice_kernel(mask: wp.array(dtype=wp.bool), start: int, step: int):
    """Set the rows selected by a normalized Python slice."""
    mask[start + wp.tid() * step] = True


@wp.kernel(enable_backward=False)
def build_per_dof_env_mask_kernel(
    indices: wp.array(dtype=wp.uint32),
    env_mask: wp.array(dtype=wp.bool),
    dof_offset: int,
    num_joints: int,
    out_mask: wp.array(dtype=wp.bool),
):
    """Build a per-DOF mask from a per-env mask, for one Newton actuator.

    Newton's :meth:`Actuator.State.reset` expects a mask of length
    ``num_actuators`` (= ``num_envs * dofs_per_actuator``). Each entry
    gates the corresponding column of the actuator's state buffers. This
    kernel maps a per-env boolean mask onto that per-DOF layout via the
    actuator's flat ``indices``.
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    out_mask[i] = env_mask[env]


@wp.kernel(enable_backward=False)
def scatter_gain_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    dof_offset: int,
    num_joints: int,
    env_stride: int,
):
    """Scatter per-actuator ``src`` values into a flat per-env-per-DOF ``dst``.

    Used by the deprecated :func:`build_newton_actuator_defaults`
    compatibility helper to scatter each ``controller.kp`` / ``controller.kd``
    into a ``(num_envs, num_joints)`` torch tensor.

    The actuator's ``indices`` are global DOF ids laid out env-major with a
    per-env stride of ``env_stride`` — the *whole model's* per-env DOF count,
    which on a floating-base articulation exceeds ``num_joints`` (the
    articulation-local, actuated joint count) by the free-root DOFs. The env
    index must therefore be decoded with ``env_stride``, not ``num_joints``;
    the articulation-local joint offset is what remains after removing the
    env's block and lands in ``[0, num_joints)`` because ``indices`` only ever
    holds this articulation's joints.

    Args:
        src: Per-actuator parameter values (e.g. ``controller.kp``).
        dst: Flat ``(num_envs * num_joints)`` articulation-local snapshot buffer.
        indices: Actuator's flat env-major global DOF indices.
        dof_offset: Offset of this articulation's DOFs in the env-major
            global index space (``0`` on PhysX, view-dependent on Newton).
        num_joints: Articulation-local joint count (``dst``'s inner stride).
        env_stride: Whole-model per-env DOF count (the stride used to build
            ``indices``).
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // env_stride
    local_dof = global_dof - env * env_stride
    dst[env * num_joints + local_dof] = src[i]


@wp.kernel(enable_backward=False)
def patch_actuator_param_kernel(
    indices: wp.array(dtype=wp.uint32),
    env_id_pos: wp.array(dtype=wp.int32),
    joint_id_pos: wp.array(dtype=wp.int32),
    values: wp.array2d(dtype=wp.float32),
    dof_offset: int,
    num_joints: int,
    dst: wp.array(dtype=wp.float32),
):
    """Per-actuator scatter for partial DR gain updates.

    For each slot ``i`` in the actuator's flat env-major ``indices``, derive
    the (env, local-joint) pair, look it up against the dense position
    arrays, and — when both axes are in the DR sub-grid — overwrite
    ``dst[i]`` (the controller parameter) with ``values[e_pos, j_pos]``.
    Cells outside the sub-grid are left untouched.

    Note:
        This kernel is PhysX-only (the Newton backend patches gains via
        :meth:`ArticulationView.set_actuator_parameter`). On PhysX every
        joint's coordinate count equals its DOF count, so the per-env stride
        used to build ``indices`` equals ``num_joints`` and the ``env`` /
        ``joint`` split below is exact. Do not reuse this kernel on a layout
        whose per-env DOF stride exceeds ``num_joints`` (e.g. a floating-base
        Newton model) without threading the true stride, or the ``joint``
        split will alias across envs — see :func:`scatter_gain_kernel`.

    Args:
        indices: Actuator's flat indices into the (env-major) DOF layout.
        env_id_pos: ``env_id_pos[env]`` gives the row in ``values`` for
            envs being updated, ``-1`` otherwise. Length ``num_envs``.
        joint_id_pos: ``joint_id_pos[joint]`` gives the column in
            ``values`` for joints being updated, ``-1`` otherwise.
            Length ``num_joints`` (articulation-local).
        values: New parameter values shaped ``(len(env_ids), len(joint_ids))``.
        dof_offset: Offset of this articulation's DOFs in the env-major
            global index space (``0`` on PhysX, view-dependent on Newton).
        num_joints: Articulation-local joint count.
        dst: Per-actuator controller parameter array (e.g. ``controller.kp``).
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    joint = global_dof % num_joints
    e_pos = env_id_pos[env]
    j_pos = joint_id_pos[joint]
    if e_pos >= 0 and j_pos >= 0:
        dst[i] = values[e_pos, j_pos]


@wp.kernel(enable_backward=False)
def patch_native_actuator_parameter(
    indices: wp.array(dtype=wp.uint32),
    canonical: wp.array2d(dtype=wp.float32),
    owner_slots: wp.array(dtype=wp.int32),
    backend_to_user: wp.array(dtype=wp.int32),
    dof_offset: int,
    num_articulation_joints: int,
    env_stride: int,
    num_envs: int,
    has_joint_ordering: bool,
    dst: wp.array(dtype=wp.float32),
):
    """Gather config-winning canonical values into one native component array."""
    i = wp.tid()
    relative_dof = int(indices[i]) - dof_offset
    if relative_dof >= 0:
        env_id = relative_dof // env_stride
        if env_id < num_envs:
            backend_joint_id = relative_dof - env_id * env_stride
            if backend_joint_id >= 0 and backend_joint_id < num_articulation_joints:
                user_joint_id = backend_joint_id
                if has_joint_ordering:
                    user_joint_id = backend_to_user[backend_joint_id]
                compact_slot = owner_slots[user_joint_id]
                if compact_slot >= 0:
                    dst[i] = canonical[env_id, compact_slot]


@wp.kernel(enable_backward=False)
def patch_native_range_parameter(
    canonical: wp.array2d(dtype=wp.float32),
    compact_joint_ids: wp.array(dtype=wp.int32),
    canonical_slots: wp.array(dtype=wp.int32),
    owner_slots: wp.array(dtype=wp.int32),
    controller_local_slots: wp.array(dtype=wp.int32),
    controller_stride: int,
    dst: wp.array(dtype=wp.float32),
):
    """Patch only final-writer canonical occurrences into one controller range."""
    env_id, range_slot = wp.tid()
    joint_id = compact_joint_ids[range_slot]
    canonical_slot = canonical_slots[range_slot]
    if owner_slots[joint_id] == canonical_slot:
        dst[env_id * controller_stride + controller_local_slots[range_slot]] = canonical[env_id, canonical_slot]


@wp.kernel(enable_backward=False)
def recompute_dc_motor_corner_velocity(
    saturation_effort: wp.array(dtype=wp.float32),
    velocity_limit: wp.array(dtype=wp.float32),
    max_motor_effort: wp.array(dtype=wp.float32),
    corner_velocity: wp.array(dtype=wp.float32),
):
    """Refresh the DC-motor derived corner velocity after a parameter write."""
    i = wp.tid()
    saturation = saturation_effort[i]
    velocity = velocity_limit[i]
    if saturation > 0.0:
        corner_velocity[i] = velocity * (1.0 + max_motor_effort[i] / saturation)
    else:
        corner_velocity[i] = velocity


@wp.kernel(enable_backward=False)
def patch_implicit_solver_parameter(
    canonical: wp.array2d(dtype=wp.float32),
    owner_slots: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_joint_ordering: bool,
    user_buffer: wp.array2d(dtype=wp.float32),
    backend_buffer: wp.array2d(dtype=wp.float32),
):
    """Gather winning implicit gains into user and Newton backend arrays."""
    env_id, user_joint_id = wp.tid()
    compact_slot = owner_slots[user_joint_id]
    if compact_slot >= 0:
        value = canonical[env_id, compact_slot]
        backend_joint_id = user_joint_id
        if has_joint_ordering:
            user_buffer[env_id, user_joint_id] = value
            backend_joint_id = user_to_backend[user_joint_id]
        backend_buffer[env_id, backend_joint_id] = value


# ---------------------------------------------------------------------------
# Articulation-level kernels: in-graph post-actuator hook.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sync_torque_telemetry(
    joint_pos_backend: wp.array2d(dtype=wp.float32),
    joint_vel_backend: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
    joint_vel_target: wp.array2d(dtype=wp.float32),
    joint_stiffness: wp.array2d(dtype=wp.float32),
    joint_damping: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    joint_modes: wp.array(dtype=wp.int32),
    native_owner: wp.array(dtype=wp.int32),
    sim_bind_joint_effort: wp.array2d(dtype=wp.float32),
    actuator_computed_effort: wp.array2d(dtype=wp.float32),
    user_to_backend: wp.array(dtype=wp.int32),
    sim_buffers_are_backend_order: bool,
    computed: wp.array2d(dtype=wp.float32),
    applied: wp.array2d(dtype=wp.float32),
):
    """In-graph post-actuator hook: fill ``computed`` / ``applied`` torque telemetry.

    For implicit DOFs we compute the shadow PD locally (no Newton actuator
    runs on these); for explicit DOFs we read the pre-clamp effort the
    actuators just scatter-added into ``actuator_computed_effort`` and the
    post-clamp effort already in ``sim_bind_joint_effort`` (= ``joint_f``).
    When the sim-bound buffers are backend-order, the live joint state
    (``joint_pos_backend`` / ``joint_vel_backend``) and both effort buffers are
    gathered through ``user_to_backend`` so every read resolves to public joint
    ``user_j``; the user-facing targets, gains, limits, and telemetry outputs are
    already user-order and are indexed at ``[i, user_j]`` directly.

    Note: ``effort_limit`` clamps only the PD shadow used for implicit-DOF
    telemetry; the FF written into ``joint_f`` is not bounded by it.
    """
    i, user_j = wp.tid()
    if native_owner[user_j] == 0:
        return
    backend_j = user_j
    if sim_buffers_are_backend_order:
        backend_j = user_to_backend[user_j]
    if joint_modes[user_j] == 1:
        err_p = joint_pos_target[i, user_j] - joint_pos_backend[i, backend_j]
        err_v = joint_vel_target[i, user_j] - joint_vel_backend[i, backend_j]
        pd = joint_stiffness[i, user_j] * err_p + joint_damping[i, user_j] * err_v
        limit = effort_limit[i, user_j]
        pd_clipped = wp.clamp(pd, -limit, limit)
        total = pd_clipped + sim_bind_joint_effort[i, backend_j]
        computed[i, user_j] = total
        applied[i, user_j] = total
    else:
        computed[i, user_j] = actuator_computed_effort[i, backend_j]
        applied[i, user_j] = sim_bind_joint_effort[i, backend_j]


def build_implicit_dof_mask(
    actuators: dict[str, ActuatorBase],
    num_joints: int,
    device: str,
    *,
    group_layouts: Sequence[_GroupLayoutLike] | None = None,
) -> tuple[wp.array, torch.Tensor]:
    """Build the compatibility per-DOF mask for implicit actuator groups.

    Entries are ``1`` for DOFs covered by an
    :class:`~isaaclab.actuators.ImplicitActuator` group, ``0`` otherwise.

    Args:
        actuators: Runtime actuator groups, used by the compatibility path.
        num_joints: Number of articulation joints.
        device: Device receiving the mask.
        group_layouts: Optional host-resident actuator group layouts. Backend
            construction passes these to build and upload the mask once.

    Returns:
        Tuple of ``(wp_mask, torch_owner)``. ``wp_mask`` is the Warp
        view used by the kernel; ``torch_owner`` is the underlying
        :class:`torch.Tensor` whose GPU memory ``wp_mask`` aliases. The
        caller **must keep a reference to** ``torch_owner`` for the
        Warp view's lifetime — otherwise the torch refcount drops to
        zero, the memory becomes eligible for reallocation by the
        caching allocator, and any captured CUDA graph that baked in
        ``wp_mask``'s device pointer will read garbage at replay time.
    """
    if group_layouts is not None:
        host_modes = [0] * num_joints
        for group in group_layouts:
            if issubclass(group.actuator_type, ImplicitActuator):
                for joint_id in group.joint_indices:
                    host_modes[joint_id] = 1
        modes = torch.tensor(host_modes, dtype=torch.int32, device=device)
        return wp.from_torch(modes, dtype=wp.int32), modes

    # Public compatibility path for callers that do not have a private
    # articulation layout. Production backends pass ``group_layouts`` and
    # therefore perform one host-built upload without device-side patching.
    modes = torch.zeros(num_joints, dtype=torch.int32, device=device)
    for actuator in actuators.values():
        if not isinstance(actuator, ImplicitActuator):
            continue
        j_ids = actuator.joint_indices
        if isinstance(j_ids, slice) or j_ids is None:
            modes[:] = 1
        else:
            modes[j_ids.long()] = 1
    return wp.from_torch(modes, dtype=wp.int32), modes


def _build_native_dof_masks(
    actuators: dict[str, ActuatorBase],
    native_group_names: frozenset[str],
    num_joints: int,
    device: str,
    *,
    group_layouts: Sequence[_GroupLayoutLike] | None = None,
) -> tuple[dict[str, wp.array], dict[str, torch.Tensor]]:
    """Build configuration-order native ownership masks for every routed field.

    A later Lab group clears an earlier native group for the fields it owns.
    The returned torch tensors retain ownership of the memory aliased by Warp.
    """
    target_fields = ("position", "velocity")
    force_fields = ("effort", "computed_effort", "applied_effort")
    if group_layouts is not None:
        target_host = [0] * num_joints
        force_host = [0] * num_joints
        touched_host = [0] * num_joints
        for group in group_layouts:
            native = group.name in native_group_names
            owns_targets = native or issubclass(group.actuator_type, ImplicitActuator)
            for joint_id in group.joint_indices:
                if native:
                    touched_host[joint_id] = 1
                if owns_targets:
                    target_host[joint_id] = int(native)
                force_host[joint_id] = int(native)
        owner_slab = torch.tensor((target_host, force_host, touched_host), dtype=torch.int32, device=device)
    else:
        # Public compatibility path for callers without host-resident layouts.
        owner_slab = torch.zeros((3, num_joints), dtype=torch.int32, device=device)
        target_owner, force_owner, touched = owner_slab
        for name, actuator in actuators.items():
            native = name in native_group_names
            owns_targets = native or isinstance(actuator, ImplicitActuator)
            joint_ids = actuator.joint_indices
            if native:
                if isinstance(joint_ids, slice) or joint_ids is None:
                    touched[:] = 1
                else:
                    touched[joint_ids.long()] = 1
            if owns_targets:
                if isinstance(joint_ids, slice) or joint_ids is None:
                    target_owner[:] = int(native)
                else:
                    target_owner[joint_ids.long()] = int(native)
            if isinstance(joint_ids, slice) or joint_ids is None:
                force_owner[:] = int(native)
            else:
                force_owner[joint_ids.long()] = int(native)

    target_owner, force_owner, touched = owner_slab
    owners = {field: target_owner for field in target_fields}
    owners.update({field: force_owner for field in force_fields})
    owners["touched"] = touched
    target_wp = wp.from_torch(target_owner, dtype=wp.int32)
    force_wp = wp.from_torch(force_owner, dtype=wp.int32)
    touched_wp = wp.from_torch(touched, dtype=wp.int32)
    masks = {field: target_wp for field in target_fields}
    masks.update({field: force_wp for field in force_fields})
    masks["touched"] = touched_wp
    return masks, owners


def build_native_dof_mask(
    actuators: dict[str, ActuatorBase],
    native_group_names: frozenset[str],
    num_joints: int,
    device: str,
) -> tuple[wp.array, torch.Tensor]:
    """Build the legacy native-DOF effort ownership mask.

    This compatibility helper preserves the former mask semantics: it marks
    every DOF belonging to a native group, irrespective of later non-native
    groups that may own the same DOF in the routed-field representation.

    Args:
        actuators: Runtime actuator groups.
        native_group_names: Names of groups backed by native Newton actuators.
        num_joints: Number of articulation joints.
        device: Device receiving the mask.

    Returns:
        Tuple of the Warp effort mask and the torch tensor that owns its
        storage.
    """
    native_actuators = {name: actuators[name] for name in native_group_names}
    masks, owners = _build_native_dof_masks(native_actuators, native_group_names, num_joints, device)
    return masks["effort"], owners["effort"]
