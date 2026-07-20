# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp ray-generation kernels for OpenCV lens-distortion cameras under the Newton renderer.

The Newton tiled camera renders by tracing an explicit per-pixel ray field of shape
``(camera_count, height, width, 2)`` (``wp.vec3f``): index ``0`` holds the ray origin in camera
space (always ``wp.vec3f(0.0)``) and index ``1`` the normalized ray direction in camera space.
Newton uses the OpenGL camera convention (``+X`` right, ``+Y`` up, looking down ``-Z``).

To honor an OpenCV ``fx/fy/cx/cy`` + distortion-coefficient calibration, for each output pixel the
kernels below invert the OpenCV forward distortion model to recover the *undistorted* normalized
image coordinates ``(x_u, y_u)`` (OpenCV image ``y`` points down), then emit the camera-space ray
``normalize(vec3(x_u, -y_u, -1))`` -- the negation of ``y`` and ``z`` maps OpenCV camera space
(``+Z`` forward, ``+Y`` down) onto Newton's OpenGL camera space.

Both kernels are launched over the ``(camera_count, height, width)`` grid with ``camera_count == 1``
(a single ray field shared across all envs; per-env transforms are applied separately by the
renderer).
"""

from __future__ import annotations

import warp as wp

# Number of fixed-point iterations used to invert the distortion models. OpenCV's own
# ``undistortPoints`` defaults to a similar iteration budget; the OpenCV pinhole inversion converges
# well within this for realistic calibrations, and the fisheye equidistant solve needs only a few.
_INVERSION_ITERATIONS = 20

# Guard against division by a near-zero radius when a pixel maps onto the optical axis.
_RADIUS_EPS = wp.constant(1.0e-12)


@wp.kernel(enable_backward=False)
def compute_camera_rays_opencv_pinhole(
    width: int,
    height: int,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    image_width: wp.float32,
    image_height: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    k5: wp.float32,
    k6: wp.float32,
    p1: wp.float32,
    p2: wp.float32,
    s1: wp.float32,
    s2: wp.float32,
    s3: wp.float32,
    s4: wp.float32,
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    """Emit camera-space rays for an OpenCV pinhole (rational + tangential + thin-prism) camera.

    The forward OpenCV model maps an undistorted normalized point ``(x, y)`` (with ``r2 = x^2 + y^2``)
    to the distorted normalized point ``(x_d, y_d)`` via a rational radial term, tangential terms
    (``p1``, ``p2``) and thin-prism terms (``s1..s4``). For each output pixel the distorted point is
    known from the pixel coordinate and the intrinsics; the kernel recovers the undistorted point by
    fixed-point iteration (matching OpenCV's :func:`undistortPoints`) and forms the ray.

    Args:
        width: Output image width [px].
        height: Output image height [px].
        fx: Focal length along the image x-axis [px].
        fy: Focal length along the image y-axis [px].
        cx: Principal point x-coordinate [px].
        cy: Principal point y-coordinate [px].
        image_width: Calibrated image width the intrinsics refer to [px].
        image_height: Calibrated image height the intrinsics refer to [px].
        k1: First radial distortion coefficient (numerator).
        k2: Second radial distortion coefficient (numerator).
        k3: Third radial distortion coefficient (numerator).
        k4: First radial distortion coefficient (denominator, rational model).
        k5: Second radial distortion coefficient (denominator, rational model).
        k6: Third radial distortion coefficient (denominator, rational model).
        p1: First tangential distortion coefficient.
        p2: Second tangential distortion coefficient.
        s1: First thin-prism distortion coefficient.
        s2: Second thin-prism distortion coefficient.
        s3: Third thin-prism distortion coefficient.
        s4: Fourth thin-prism distortion coefficient.
        out_rays: Ray field of shape ``(1, height, width, 2)``: ``[..., 0]`` origin, ``[..., 1]``
            direction, both in Newton's OpenGL camera space.
    """
    camera_index, py, px = wp.tid()

    # Map the render pixel onto the calibrated image grid, then to distorted normalized coordinates.
    # OpenCV image y points down.
    u = ((wp.float32(px) + 0.5) / wp.float32(width)) * image_width
    v = ((wp.float32(py) + 0.5) / wp.float32(height)) * image_height
    x_d = (u - cx) / fx
    y_d = (v - cy) / fy

    # Fixed-point inversion of the forward model, seeded at the distorted point.
    x = x_d
    y = y_d
    for _i in range(_INVERSION_ITERATIONS):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4
        x = (x_d - dx) / radial
        y = (y_d - dy) / radial

    # OpenCV camera space (+Z forward, +Y down) -> Newton OpenGL camera space (-Z forward, +Y up).
    ray_direction_camera_space = wp.normalize(wp.vec3f(x, -y, -1.0))
    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space


@wp.kernel(enable_backward=False)
def compute_camera_rays_opencv_fisheye(
    width: int,
    height: int,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    image_width: wp.float32,
    image_height: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    """Emit camera-space rays for an OpenCV fisheye (equidistant ``k1..k4``) camera.

    The OpenCV fisheye forward model maps an incidence angle ``theta`` to a distorted radius
    ``theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``. For each output
    pixel ``theta_d`` is the radius of the distorted normalized point; the kernel recovers ``theta``
    by fixed-point iteration, then scales the undistorted radius by ``tan(theta)`` and forms the ray.

    Args:
        width: Output image width [px].
        height: Output image height [px].
        fx: Focal length along the image x-axis [px].
        fy: Focal length along the image y-axis [px].
        cx: Principal point x-coordinate [px].
        cy: Principal point y-coordinate [px].
        image_width: Calibrated image width the intrinsics refer to [px].
        image_height: Calibrated image height the intrinsics refer to [px].
        k1: First fisheye distortion coefficient.
        k2: Second fisheye distortion coefficient.
        k3: Third fisheye distortion coefficient.
        k4: Fourth fisheye distortion coefficient.
        out_rays: Ray field of shape ``(1, height, width, 2)``: ``[..., 0]`` origin, ``[..., 1]``
            direction, both in Newton's OpenGL camera space.
    """
    camera_index, py, px = wp.tid()

    u = ((wp.float32(px) + 0.5) / wp.float32(width)) * image_width
    v = ((wp.float32(py) + 0.5) / wp.float32(height)) * image_height
    x_d = (u - cx) / fx
    y_d = (v - cy) / fy

    theta_d = wp.sqrt(x_d * x_d + y_d * y_d)

    # On the optical axis the direction is straight ahead in Newton's OpenGL camera space.
    if theta_d < _RADIUS_EPS:
        out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
        out_rays[camera_index, py, px, 1] = wp.vec3f(0.0, 0.0, -1.0)
        return

    # Fixed-point inversion of theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8), seeded at theta_d.
    theta = theta_d
    for _i in range(_INVERSION_ITERATIONS):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t6 * t2
        radial = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
        theta = theta_d / radial

    # Undistorted normalized radius from the recovered incidence angle.
    r_u = wp.tan(theta)
    scale = r_u / theta_d
    x_u = x_d * scale
    y_u = y_d * scale

    ray_direction_camera_space = wp.normalize(wp.vec3f(x_u, -y_u, -1.0))
    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space
