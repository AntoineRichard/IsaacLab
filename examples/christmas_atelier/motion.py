# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted waypoints and independent geometric measurements for the atelier."""

import numpy as np
import trimesh

from .assets import CANDY_CAPSULE_HALF_LENGTH, CANDY_CORE_HALF_LENGTH, CANDY_CORE_RADIUS, candy_mesh


def choreography(pick_positions: list[tuple[float, float, float]], mouth: tuple[float, float, float]) -> tuple:
    """Return arrival times, TCP positions, finger openings and named phases.

    Holds before grasping and after releasing allow contact to settle. Transfer occurs above
    both bin walls and the cuff. Each motion segment uses minimum-jerk interpolation.
    """
    home = (0.38, -0.02, 1.22)
    times, positions, fingers, phases = [0.0], [home], [0.04], ["home"]

    def arrive(duration: float, position: tuple, opening: float, phase: str) -> None:
        times.append(times[-1] + duration)
        positions.append(position)
        fingers.append(opening)
        phases.append(phase)

    arrive(1.0, home, 0.04, "settle")
    for index, pick in enumerate(pick_positions):
        above_bin = (pick[0], pick[1], 1.25)
        above_sock = (mouth[0], mouth[1], mouth[2] + 0.15)
        release = (mouth[0], mouth[1], mouth[2] + 0.075)
        arrive(2.0, above_bin, 0.04, f"approach_{index}")
        arrive(2.0, pick, 0.04, f"descend_{index}")
        arrive(0.8, pick, 0.010, f"grasp_{index}")
        arrive(2.0, above_bin, 0.010, f"lift_{index}")
        arrive(2.5, above_sock, 0.010, f"transport_{index}")
        arrive(1.0, release, 0.010, f"lower_{index}")
        arrive(0.8, release, 0.04, f"release_{index}")
        arrive(0.8, release, 0.04, f"drop_{index}")
        arrive(1.0, above_sock, 0.04, f"retreat_{index}")
    arrive(2.0, home, 0.04, "return")
    arrive(3.0, home, 0.04, "finished")
    return np.array(times), np.array(positions), np.array(fingers), phases


def sample_motion(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate targets with zero velocity and acceleration at each waypoint."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
    return targets[segment] + blend * (targets[segment + 1] - targets[segment])


def _closed_shell(vertices: np.ndarray, faces: np.ndarray, cuff: np.ndarray) -> np.ndarray:
    """Cap the supported opening for measurement only, without changing the physics mesh."""
    capped = np.vstack([vertices, vertices[cuff].mean(axis=0)])
    cap = [(len(vertices), int(cuff[i]), int(cuff[(i + 1) % len(cuff)])) for i in range(len(cuff))]
    return capped[np.vstack([faces, cap])]


def _inside_shell(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Evaluate solid-angle winding in bounded batches for a deformed closed shell."""
    inside = []
    for start in range(0, len(points), 64):
        rays = triangles[None, :, :, :] - points[start : start + 64, None, None, :]
        a, b, c = rays[:, :, 0], rays[:, :, 1], rays[:, :, 2]
        la, lb, lc = (np.linalg.norm(vector, axis=-1) for vector in (a, b, c))
        denominator = (
            la * lb * lc + np.sum(a * b, axis=-1) * lc + np.sum(b * c, axis=-1) * la + np.sum(c * a, axis=-1) * lb
        )
        angles = 2 * np.arctan2(np.sum(a * np.cross(b, c), axis=-1), denominator)
        inside.extend(np.abs(angles.sum(axis=1)) / (4 * np.pi) > 0.9)
    return np.asarray(inside, dtype=bool)


def _clip_polygon(polygon: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Clip a convex polygon to the half-space of nonpositive signed distances."""
    clipped = []
    start, start_distance = polygon[-1], distances[-1]
    for end, end_distance in zip(polygon, distances, strict=True):
        if start_distance * end_distance < 0.0:
            fraction = start_distance / (start_distance - end_distance)
            clipped.append(start + fraction * (end - start))
        if end_distance <= 0.0:
            clipped.append(end)
        start, start_distance = end, end_distance
    return np.asarray(clipped)


def _intersects_cylinder(triangle: np.ndarray, axis: np.ndarray) -> bool:
    """Test a shell triangle against the cylinder centered at the origin."""
    # Restrict the triangle to the space between the two flat end caps.
    polygon = _clip_polygon(triangle, triangle @ axis - CANDY_CORE_HALF_LENGTH)
    if not len(polygon):
        return False
    polygon = _clip_polygon(polygon, -polygon @ axis - CANDY_CORE_HALF_LENGTH)
    if not len(polygon):
        return False

    # The remaining polygon intersects the cylinder if its projection intersects the disk.
    radial = polygon - (polygon @ axis)[:, None] * axis
    following = np.roll(radial, -1, axis=0)
    cross = np.cross(radial, following) @ axis
    if abs(cross.sum()) > 1.0e-14 and (np.all(cross >= 0.0) or np.all(cross <= 0.0)):
        return True  # The projected polygon contains the cylinder axis.
    edges = following - radial
    edge_lengths_squared = np.sum(edges**2, axis=1)
    nearest_fraction = np.divide(
        -np.sum(radial * edges, axis=1),
        edge_lengths_squared,
        out=np.zeros(len(edges)),
        where=edge_lengths_squared > 0.0,
    )
    nearest = radial + np.clip(nearest_fraction, 0.0, 1.0)[:, None] * edges
    return bool(np.any(np.sum(nearest**2, axis=1) <= CANDY_CORE_RADIUS**2))


def candy_is_contained(
    center: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cuff: np.ndarray,
    rotation: np.ndarray | None = None,
    *,
    candy_collider: str = "capsule",
) -> bool:
    """Test the oriented candy core against the current cloth shell.

    Test the finite cylinder exactly or bound the capsule with centerline clearance samples.
    The supported opening is capped only for this measurement.
    """
    triangles = _closed_shell(vertices, faces, cuff) - center
    if not _inside_shell(np.zeros((1, 3)), triangles)[0]:
        return False
    axis = np.array([1.0, 0.0, 0.0])
    if rotation is not None:
        vector, scalar = rotation[:3], rotation[3]
        axis += 2 * np.cross(vector, np.cross(vector, axis) + scalar * axis)
        axis /= np.linalg.norm(axis)
    if candy_collider == "capsule":
        points = np.linspace(-CANDY_CAPSULE_HALF_LENGTH, CANDY_CAPSULE_HALF_LENGTH, 5)[:, None] * axis
        if not np.all(_inside_shell(points, triangles)):
            return False
        repeated = np.repeat(points, len(triangles), axis=0)
        closest = trimesh.triangles.closest_point(np.tile(triangles, (len(points), 1, 1)), repeated)
        clearance = CANDY_CORE_RADIUS + CANDY_CAPSULE_HALF_LENGTH / 4
        return bool(np.linalg.norm(closest - repeated, axis=1).min() > clearance)
    extent = CANDY_CORE_HALF_LENGTH * np.abs(axis) + CANDY_CORE_RADIUS * np.sqrt(np.maximum(1.0 - axis**2, 0.0))
    nearby = np.all((triangles.min(axis=1) <= extent) & (triangles.max(axis=1) >= -extent), axis=1)
    return not any(_intersects_cylinder(triangle, axis) for triangle in triangles[nearby])


def measure_stocking(
    vertices: np.ndarray,
    rest: np.ndarray,
    faces: np.ndarray,
    cuff: np.ndarray,
    candies: np.ndarray,
    *,
    candy_collider: str = "capsule",
) -> dict:
    """Measure live cloth integrity and candy containment, preserving JSON on solver divergence."""
    finite = bool(np.isfinite(vertices).all())
    result = {
        "cloth_finite": finite,
        "cuff_drift_m": None,
        "max_cloth_edge_stretch_ratio": None,
        "contained": [False] * len(candies),
        "wrapper_contained": [False] * len(candies),
        "cloth_rms_displacement_m": None,
    }
    if not finite:
        return result
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    result["cuff_drift_m"] = float(np.linalg.norm(vertices[cuff] - rest[cuff], axis=1).max())
    result["max_cloth_edge_stretch_ratio"] = float(np.max(lengths / rest_lengths))
    result["cloth_rms_displacement_m"] = float(np.sqrt(np.mean(np.sum((vertices - rest) ** 2, axis=1))))
    result["contained"] = [
        bool(np.isfinite(pose).all())
        and candy_is_contained(pose[:3], vertices, faces, cuff, pose[3:], candy_collider=candy_collider)
        for pose in candies
    ]
    if len(candies):
        wrapper_vertices, wrapper_faces, _ = candy_mesh()
        wrapper_edges = np.unique(
            np.sort(np.vstack([wrapper_faces[:, [0, 1]], wrapper_faces[:, [1, 2]], wrapper_faces[:, [2, 0]]]), axis=1),
            axis=0,
        )
        samples = np.vstack(
            [
                wrapper_vertices,
                wrapper_vertices[wrapper_edges].mean(axis=1),
                wrapper_vertices[wrapper_faces].mean(axis=1),
            ]
        )
        triangles = _closed_shell(vertices, faces, cuff)
        for index, pose in enumerate(candies):
            if not np.isfinite(pose).all():
                continue
            vector, scalar = pose[3:6], pose[6]
            world = samples + 2 * np.cross(vector, np.cross(vector, samples) + scalar * samples) + pose[:3]
            result["wrapper_contained"][index] = bool(np.all(_inside_shell(world, triangles)))
    return result
