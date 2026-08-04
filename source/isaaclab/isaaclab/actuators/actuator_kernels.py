# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels used by actuator collections."""

from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher


@wp.kernel(enable_backward=False)
def write_2d_float_with_indices(
    source: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    full_data: bool,
    target: wp.array2d(dtype=wp.float32),
):
    """Write 2-D float data into a target buffer using environment and joint indices."""
    env_i, joint_i = wp.tid()
    env_id = env_ids[env_i]
    joint_id = joint_ids[joint_i]
    if full_data:
        target[env_id, joint_id] = source[env_id, joint_id]
    else:
        target[env_id, joint_id] = source[env_i, joint_i]


_WRITE_2D_FLOAT_WITH_INDICES_DISPATCHER = IndexKernelDispatcher(write_2d_float_with_indices, ("env_ids", "joint_ids"))


def write_2d_float_with_indices_kernel(
    env_ids: torch.Tensor | wp.array, joint_ids: torch.Tensor | wp.array
) -> wp.Kernel:
    """Select the indexed float writer for the selector dtypes.

    Args:
        env_ids: Environment indices.
        joint_ids: Joint indices.

    Returns:
        Warp kernel specialized for the two selector dtypes.
    """
    return _WRITE_2D_FLOAT_WITH_INDICES_DISPATCHER.select(env_ids, joint_ids)


@wp.kernel(enable_backward=False)
def write_2d_float_with_mask(
    source: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    target: wp.array2d(dtype=wp.float32),
):
    """Write full-sized 2-D float data into a target buffer using masks."""
    env_id, joint_id = wp.tid()
    if env_mask[env_id] and joint_mask[joint_id]:
        target[env_id, joint_id] = source[env_id, joint_id]


@wp.kernel(enable_backward=False)
def record_last_scoped_parameter_position(
    ids: wp.array(dtype=Any),
    upper_bound: int,
    last_positions: wp.array(dtype=wp.int32),
):
    """Record the final selected position for each bounded signed selector id."""
    position = wp.tid()
    selector_id = ids[position]
    if selector_id >= 0 and selector_id < upper_bound:
        wp.atomic_max(last_positions, selector_id, position)


@wp.kernel(enable_backward=False)
def write_scoped_parameter_index(
    source: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    scope_joint_ids: wp.array(dtype=wp.int32),
    csr_offsets: wp.array(dtype=wp.int32),
    csr_slots: wp.array(dtype=wp.int32),
    group_inverse: wp.array(dtype=wp.int32),
    last_env_positions: wp.array(dtype=wp.int32),
    last_joint_positions: wp.array(dtype=wp.int32),
    num_worlds: int,
    num_articulation_joints: int,
    explicit_joint_ids: bool,
    type_scope: bool,
    group_scope: bool,
    value_mode: int,
    target: wp.array2d(dtype=wp.float32),
):
    """Write a scoped parameter using Cartesian articulation index selectors.

    The final dimension is the stable compact slot. Every candidate thread checks
    later Cartesian entries for the same destination, making duplicate selectors
    deterministic without atomics on both CPU and CUDA.
    """
    env_i, joint_i, candidate_i = wp.tid()
    env_id = env_ids[env_i]
    if env_id < 0 or env_id >= num_worlds:
        return

    if explicit_joint_ids:
        articulation_joint_id = joint_ids[joint_i]
        if articulation_joint_id < 0 or articulation_joint_id >= num_articulation_joints:
            return
        if type_scope:
            csr_joint_id = wp.int32(articulation_joint_id)
            compact_i = csr_offsets[csr_joint_id] + candidate_i
            if compact_i >= csr_offsets[csr_joint_id + wp.int32(1)]:
                return
            slot = csr_slots[compact_i]
        elif group_scope:
            slot = group_inverse[wp.int32(articulation_joint_id)]
            if slot < 0:
                return
        else:
            slot = candidate_i
            if scope_joint_ids[slot] != articulation_joint_id:
                return
        if last_env_positions[env_id] != env_i or last_joint_positions[wp.int32(articulation_joint_id)] != joint_i:
            return
    else:
        if candidate_i != 0:
            return
        slot = joint_i
        if last_env_positions[env_id] != env_i:
            return

    if value_mode == 0:
        target[env_id, slot] = source[0, 0]
    elif value_mode == 1:
        target[env_id, slot] = source[0, joint_i]
    else:
        target[env_id, slot] = source[env_i, joint_i]


@wp.kernel(enable_backward=False)
def write_scoped_parameter_mask(
    source: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    scope_joint_ids: wp.array(dtype=wp.int32),
    value_mode: int,
    target: wp.array2d(dtype=wp.float32),
):
    """Write a scoped parameter using full-articulation masks and compact values."""
    env_id, slot = wp.tid()
    if not env_mask[env_id] or not joint_mask[scope_joint_ids[slot]]:
        return
    if value_mode == 0:
        target[env_id, slot] = source[0, 0]
    elif value_mode == 1:
        target[env_id, slot] = source[0, slot]
    else:
        target[env_id, slot] = source[env_id, slot]


@wp.kernel(enable_backward=False)
def scatter_processed_targets(
    source_pos: wp.array2d(dtype=wp.float32),
    source_vel: wp.array2d(dtype=wp.float32),
    source_effort: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    target_pos: wp.array2d(dtype=wp.float32),
    target_vel: wp.array2d(dtype=wp.float32),
    target_effort: wp.array2d(dtype=wp.float32),
):
    """Scatter actuator command outputs into full articulation command buffers.

    Only non-None source arrays are processed. Explicit actuator models (for example
    :class:`~isaaclab.actuators.IdealPDActuator`) clear the position and velocity
    commands after computing the effort, so those sources may be null.
    """
    env_id, source_joint_id = wp.tid()
    target_joint_id = joint_indices[source_joint_id]
    if source_pos:
        target_pos[env_id, target_joint_id] = source_pos[env_id, source_joint_id]
    if source_vel:
        target_vel[env_id, target_joint_id] = source_vel[env_id, source_joint_id]
    if source_effort:
        target_effort[env_id, target_joint_id] = source_effort[env_id, source_joint_id]


@wp.kernel(enable_backward=False)
def scatter_actuator_state_model(
    source_computed_effort: wp.array2d(dtype=wp.float32),
    source_applied_effort: wp.array2d(dtype=wp.float32),
    source_gear_ratio: wp.array2d(dtype=wp.float32),
    source_velocity_limit: wp.array2d(dtype=wp.float32),
    has_gear_ratio: bool,
    joint_indices: wp.array(dtype=wp.int32),
    target_computed_effort: wp.array2d(dtype=wp.float32),
    target_applied_effort: wp.array2d(dtype=wp.float32),
    target_gear_ratio: wp.array2d(dtype=wp.float32),
    target_velocity_limit: wp.array2d(dtype=wp.float32),
):
    """Scatter actuator telemetry into full articulation actuator buffers."""
    env_id, source_joint_id = wp.tid()
    target_joint_id = joint_indices[source_joint_id]
    target_computed_effort[env_id, target_joint_id] = source_computed_effort[env_id, source_joint_id]
    target_applied_effort[env_id, target_joint_id] = source_applied_effort[env_id, source_joint_id]
    if has_gear_ratio:
        target_gear_ratio[env_id, target_joint_id] = source_gear_ratio[env_id, source_joint_id]
    target_velocity_limit[env_id, target_joint_id] = source_velocity_limit[env_id, source_joint_id]


@wp.kernel(enable_backward=False)
def compute_implicit_actuator_batch(
    command_pos: wp.array2d(dtype=wp.float32),
    command_vel: wp.array2d(dtype=wp.float32),
    command_effort: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    velocity_limit: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    batch_computed_effort: wp.array2d(dtype=wp.float32),
    batch_applied_effort: wp.array2d(dtype=wp.float32),
    target_pos: wp.array2d(dtype=wp.float32),
    target_vel: wp.array2d(dtype=wp.float32),
    target_effort: wp.array2d(dtype=wp.float32),
    computed_effort: wp.array2d(dtype=wp.float32),
    applied_effort: wp.array2d(dtype=wp.float32),
    soft_velocity_limit: wp.array2d(dtype=wp.float32),
):
    """Compute and publish one implicit actuator execution batch."""
    env_id, batch_joint_id = wp.tid()
    joint_id = joint_indices[batch_joint_id]

    position_target = command_pos[env_id, joint_id]
    velocity_target = command_vel[env_id, joint_id]
    feedforward = command_effort[env_id, joint_id]
    effort = (
        stiffness[env_id, batch_joint_id] * (position_target - joint_pos[env_id, joint_id])
        + damping[env_id, batch_joint_id] * (velocity_target - joint_vel[env_id, joint_id])
        + feedforward
    )
    limit = effort_limit[env_id, batch_joint_id]
    clamped_effort = wp.clamp(effort, -limit, limit)

    batch_computed_effort[env_id, batch_joint_id] = effort
    batch_applied_effort[env_id, batch_joint_id] = clamped_effort
    target_pos[env_id, joint_id] = position_target
    target_vel[env_id, joint_id] = velocity_target
    target_effort[env_id, joint_id] = feedforward
    computed_effort[env_id, joint_id] = effort
    applied_effort[env_id, joint_id] = clamped_effort
    soft_velocity_limit[env_id, joint_id] = velocity_limit[env_id, batch_joint_id]


@wp.kernel(enable_backward=False)
def gather_actuator_batch(
    command_pos: wp.array2d(dtype=wp.float32),
    command_vel: wp.array2d(dtype=wp.float32),
    command_effort: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    batch_command_pos: wp.array2d(dtype=wp.float32),
    batch_command_vel: wp.array2d(dtype=wp.float32),
    batch_command_effort: wp.array2d(dtype=wp.float32),
    batch_joint_pos: wp.array2d(dtype=wp.float32),
    batch_joint_vel: wp.array2d(dtype=wp.float32),
):
    """Gather full articulation commands and state into one execution batch."""
    env_id, batch_joint_id = wp.tid()
    joint_id = joint_indices[batch_joint_id]
    batch_command_pos[env_id, batch_joint_id] = command_pos[env_id, joint_id]
    batch_command_vel[env_id, batch_joint_id] = command_vel[env_id, joint_id]
    batch_command_effort[env_id, batch_joint_id] = command_effort[env_id, joint_id]
    batch_joint_pos[env_id, batch_joint_id] = joint_pos[env_id, joint_id]
    batch_joint_vel[env_id, batch_joint_id] = joint_vel[env_id, joint_id]
