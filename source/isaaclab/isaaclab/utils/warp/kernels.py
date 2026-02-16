# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom kernels for warp."""

from typing import Any

import warp as wp

##
# Raycasting
##


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    max_dist: float = 1e6,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
):
    """Performs ray-casting against a mesh.

    This function performs ray-casting against the given mesh using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    Note that the `ray_starts`, `ray_directions`, and `ray_hits` arrays should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    The function utilizes the `mesh_query_ray` method from the `wp` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to `1e6` units.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (N, 3).
        ray_directions: The input ray directions. Shape is (N, 3).
        ray_hits: The output ray hit positions. Shape is (N, 3).
        ray_distance: The output ray hit distances. Shape is (N,), if `return_distance` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (N, 3), if `return_normal` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (N,), if `return_face_id` is True. Otherwise,
            this array is not used.
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_distance: Whether to return the ray hit distances. Defaults to False.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
    """
    # get the thread id
    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # ray cast against the mesh and store the hit position
    hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)
    # if the ray hit, store the hit data
    if hit_success:
        ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]
        if return_distance == 1:
            ray_distance[tid] = t
        if return_normal == 1:
            ray_normal[tid] = n
        if return_face_id == 1:
            ray_face_id[tid] = f


@wp.kernel(enable_backward=False)
def raycast_static_meshes_kernel(
    mesh: wp.array2d(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    ray_hits: wp.array2d(dtype=wp.vec3),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
    max_dist: float = 1e6,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
):
    """Performs ray-casting against multiple static meshes.

    This function performs ray-casting against the given meshes using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    The function utilizes the ``mesh_query_ray`` method from the ``wp`` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to ``1e6`` units.

    .. note::
        That the ``ray_starts``, ``ray_directions``, and ``ray_hits`` arrays should have compatible shapes
        and data types to ensure proper execution. Additionally, they all must be in the same frame.

        This kernel differs from the :meth:`raycast_dynamic_meshes_kernel` in that it does not take into
        account the mesh's position and rotation. This kernel is useful for ray-casting against static meshes
        that are not expected to move.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The output ray hit distances. Shape is (B, N,), if ``return_distance`` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
        return_mesh_id: Whether to return the mesh id. Defaults to False.
    """
    # get the thread id
    tid_mesh_id, tid_env, tid_ray = wp.tid()

    direction = ray_directions[tid_env, tid_ray]
    start_pos = ray_starts[tid_env, tid_ray]

    # ray cast against the mesh and store the hit position
    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)

    # if the ray hit, store the hit data
    if mesh_query_ray_t.result:
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # check if hit distance is less than the current hit distance, only then update the memory
        # TODO, in theory we could use the output of atomic_min to avoid the non-thread safe next comparison
        # however, warp atomic_min is returning the wrong values on gpu currently.
        # FIXME https://github.com/NVIDIA/warp/issues/1058
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            # convert back to world space and update the hit data
            ray_hits[tid_env, tid_ray] = start_pos + mesh_query_ray_t.t * direction

            # update the normal and face id if requested
            if return_normal == 1:
                ray_normal[tid_env, tid_ray] = mesh_query_ray_t.normal
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)


@wp.kernel(enable_backward=False)
def raycast_dynamic_meshes_kernel(
    mesh: wp.array2d(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    ray_hits: wp.array2d(dtype=wp.vec3),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
    mesh_positions: wp.array2d(dtype=wp.vec3),
    mesh_rotations: wp.array2d(dtype=wp.quat),
    max_dist: float = 1e6,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
):
    """Performs ray-casting against multiple meshes.

    This function performs ray-casting against the given meshes using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    The function utilizes the ``mesh_query_ray`` method from the ``wp`` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to ``1e6`` units.


    Note:
        That the ``ray_starts``, ``ray_directions``, and ``ray_hits`` arrays should have compatible shapes
        and data types to ensure proper execution. Additionally, they all must be in the same frame.

        All arguments are expected to be batched with the first dimension (B, batch) being the number of envs
        and the second dimension (N, num_rays) being the number of rays. For Meshes, W is the number of meshes.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The output ray hit distances. Shape is (B, N,), if ``return_distance`` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        mesh_positions: The input mesh positions in world frame. Shape is (W, 3).
        mesh_rotations: The input mesh rotations in world frame. Shape is (W, 4).
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
        return_mesh_id: Whether to return the mesh id. Defaults to False.
    """
    # get the thread id
    tid_mesh_id, tid_env, tid_ray = wp.tid()

    mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])
    start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])

    # ray cast against the mesh and store the hit position
    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)
    # if the ray hit, store the hit data
    if mesh_query_ray_t.result:
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # check if hit distance is less than the current hit distance, only then update the memory
        # TODO, in theory we could use the output of atomic_min to avoid the non-thread safe next comparison
        # however, warp atomic_min is returning the wrong values on gpu currently.
        # FIXME https://github.com/NVIDIA/warp/issues/1058
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            # convert back to world space and update the hit data
            hit_pos = start_pos + mesh_query_ray_t.t * direction
            ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)

            # update the normal and face id if requested
            if return_normal == 1:
                n = wp.transform_vector(mesh_pose, mesh_query_ray_t.normal)
                ray_normal[tid_env, tid_ray] = n
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)


@wp.kernel(enable_backward=False)
def reshape_tiled_image(
    tiled_image_buffer: Any,
    batched_image: Any,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_tiles_x: int,
):
    """Reshapes a tiled image into a batch of images.

    This function reshapes the input tiled image buffer into a batch of images. The input image buffer
    is assumed to be tiled in the x and y directions. The output image is a batch of images with the
    specified height, width, and number of channels.

    Args:
        tiled_image_buffer: The input image buffer. Shape is (height * width * num_channels * num_cameras,).
        batched_image: The output image. Shape is (num_cameras, height, width, num_channels).
        image_width: The width of the image.
        image_height: The height of the image.
        num_channels: The number of channels in the image.
        num_tiles_x: The number of tiles in x-direction.
    """
    # get the thread id
    camera_id, height_id, width_id = wp.tid()

    # resolve the tile indices
    tile_x_id = camera_id % num_tiles_x
    tile_y_id = camera_id // num_tiles_x
    # compute the start index of the pixel in the tiled image buffer
    pixel_start = (
        num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )

    # copy the pixel values into the batched image
    for i in range(num_channels):
        batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])


# uint32 -> int32 conversion is required for non-colored segmentation annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.uint32), "batched_image": wp.array(dtype=wp.uint32, ndim=4)},
)
# uint8 is used for 4 channel annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.uint8), "batched_image": wp.array(dtype=wp.uint8, ndim=4)},
)
# float32 is used for single channel annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.float32), "batched_image": wp.array(dtype=wp.float32, ndim=4)},
)

##
# Wrench Composer
##


@wp.kernel
def set_forces_to_dual_buffers(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    link_positions: wp.array2d(dtype=wp.vec3f),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Sets forces and torques into the appropriate global or local buffer.

    Routes forces/torques to either global (world-frame) or local (body-frame) buffers
    based on the ``is_global`` flag. Position offsets contribute torque via cross product.

    Args:
        env_ids: Environment indices.
        body_ids: Body indices.
        forces: Input forces. Shape: (len(env_ids), len(body_ids)).
        torques: Input torques. Shape: (len(env_ids), len(body_ids)).
        positions: Application positions. Shape: (len(env_ids), len(body_ids)).
        link_positions: Link frame positions (only used when is_global=True with positions).
        global_force_w: Output global force buffer. Shape: (num_envs, num_bodies).
        global_torque_w: Output global torque buffer. Shape: (num_envs, num_bodies).
        local_force_b: Output local force buffer. Shape: (num_envs, num_bodies).
        local_torque_b: Output local torque buffer. Shape: (num_envs, num_bodies).
        is_global: Whether forces/torques are in the global frame.
    """
    tid_env, tid_body = wp.tid()
    ei = env_ids[tid_env]
    bi = body_ids[tid_body]

    if is_global:
        if torques:
            global_torque_w[ei, bi] = torques[tid_env, tid_body]
        if forces:
            global_force_w[ei, bi] = forces[tid_env, tid_body]
            if positions:
                offset = positions[tid_env, tid_body] - link_positions[ei, bi]
                global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(offset, forces[tid_env, tid_body])
    else:
        if torques:
            local_torque_b[ei, bi] = torques[tid_env, tid_body]
        if forces:
            local_force_b[ei, bi] = forces[tid_env, tid_body]
            if positions:
                local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(
                    positions[tid_env, tid_body], forces[tid_env, tid_body]
                )


@wp.kernel
def add_forces_to_dual_buffers(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    link_positions: wp.array2d(dtype=wp.vec3f),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Adds forces and torques into the appropriate global or local buffer.

    Same as :func:`set_forces_to_dual_buffers` but uses ``+=`` instead of ``=``.

    Args:
        env_ids: Environment indices.
        body_ids: Body indices.
        forces: Input forces. Shape: (len(env_ids), len(body_ids)).
        torques: Input torques. Shape: (len(env_ids), len(body_ids)).
        positions: Application positions. Shape: (len(env_ids), len(body_ids)).
        link_positions: Link frame positions (only used when is_global=True with positions).
        global_force_w: Output global force buffer. Shape: (num_envs, num_bodies).
        global_torque_w: Output global torque buffer. Shape: (num_envs, num_bodies).
        local_force_b: Output local force buffer. Shape: (num_envs, num_bodies).
        local_torque_b: Output local torque buffer. Shape: (num_envs, num_bodies).
        is_global: Whether forces/torques are in the global frame.
    """
    tid_env, tid_body = wp.tid()
    ei = env_ids[tid_env]
    bi = body_ids[tid_body]

    if is_global:
        if forces:
            global_force_w[ei, bi] = global_force_w[ei, bi] + forces[tid_env, tid_body]
            if positions:
                offset = positions[tid_env, tid_body] - link_positions[ei, bi]
                global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(offset, forces[tid_env, tid_body])
        if torques:
            global_torque_w[ei, bi] = global_torque_w[ei, bi] + torques[tid_env, tid_body]
    else:
        if forces:
            local_force_b[ei, bi] = local_force_b[ei, bi] + forces[tid_env, tid_body]
            if positions:
                local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(
                    positions[tid_env, tid_body], forces[tid_env, tid_body]
                )
        if torques:
            local_torque_b[ei, bi] = local_torque_b[ei, bi] + torques[tid_env, tid_body]


@wp.kernel
def add_raw_wrench_buffers(
    src_gf: wp.array2d(dtype=wp.vec3f),
    src_gt: wp.array2d(dtype=wp.vec3f),
    src_lf: wp.array2d(dtype=wp.vec3f),
    src_lt: wp.array2d(dtype=wp.vec3f),
    dst_gf: wp.array2d(dtype=wp.vec3f),
    dst_gt: wp.array2d(dtype=wp.vec3f),
    dst_lf: wp.array2d(dtype=wp.vec3f),
    dst_lt: wp.array2d(dtype=wp.vec3f),
):
    """Element-wise adds source wrench buffers into destination buffers.

    Used to merge one composer's 4 buffers (global force/torque, local force/torque)
    into another composer's buffers.

    Args:
        src_gf: Source global forces.
        src_gt: Source global torques.
        src_lf: Source local forces.
        src_lt: Source local torques.
        dst_gf: Destination global forces (modified in-place).
        dst_gt: Destination global torques (modified in-place).
        dst_lf: Destination local forces (modified in-place).
        dst_lt: Destination local torques (modified in-place).
    """
    tid_env, tid_body = wp.tid()
    dst_gf[tid_env, tid_body] = dst_gf[tid_env, tid_body] + src_gf[tid_env, tid_body]
    dst_gt[tid_env, tid_body] = dst_gt[tid_env, tid_body] + src_gt[tid_env, tid_body]
    dst_lf[tid_env, tid_body] = dst_lf[tid_env, tid_body] + src_lf[tid_env, tid_body]
    dst_lt[tid_env, tid_body] = dst_lt[tid_env, tid_body] + src_lt[tid_env, tid_body]


@wp.kernel
def compose_wrench_to_body_frame(
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    link_quaternions: wp.array2d(dtype=wp.quatf),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Composes global and local wrench buffers into a single body-frame output.

    Rotates global-frame forces/torques into body frame using ``quat_rotate_inv``,
    then adds the local-frame forces/torques.

    Args:
        global_force_w: Global forces in world frame.
        global_torque_w: Global torques in world frame.
        local_force_b: Local forces in body frame.
        local_torque_b: Local torques in body frame.
        link_quaternions: Body quaternions (xyzw convention for warp).
        out_force_b: Output composed force in body frame.
        out_torque_b: Output composed torque in body frame.
    """
    tid_env, tid_body = wp.tid()
    q = link_quaternions[tid_env, tid_body]
    out_force_b[tid_env, tid_body] = wp.quat_rotate_inv(q, global_force_w[tid_env, tid_body]) + local_force_b[
        tid_env, tid_body
    ]
    out_torque_b[tid_env, tid_body] = wp.quat_rotate_inv(q, global_torque_w[tid_env, tid_body]) + local_torque_b[
        tid_env, tid_body
    ]
