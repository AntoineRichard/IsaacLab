# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible, meter-scale stocking and wrapped-candy meshes; no external art tools required."""

import numpy as np
from scipy.interpolate import PchipInterpolator

RED = (0.68, 0.025, 0.045)
CREAM = (0.96, 0.90, 0.75)
GREEN = (0.025, 0.30, 0.13)
GOLD = (0.95, 0.60, 0.08)
CANDY_CORE_RADIUS = 0.014
CANDY_CORE_HALF_LENGTH = 0.014
CANDY_CAPSULE_HALF_LENGTH = 0.005


def stocking_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shape two cloth panels around an open cuff and join their side and sole seams.

    Rows run across the garment. The two side outlines independently shape the heel and
    instep. The curved sole closes on a line of shared vertices, as on a sewn stocking.
    """
    # Coordinates in the flat pattern, from cuff to sole. Pin the mouth to the filling hoop.
    stations = [0.0, 0.16, 0.46, 0.62, 0.73, 0.81, 0.88, 0.95, 1.0]
    heel = PchipInterpolator(
        stations,
        [
            [-0.075, 0],
            [-0.091, -0.058],
            [-0.084, -0.158],
            [-0.082, -0.215],
            [-0.087, -0.250],
            [-0.087, -0.273],
            [-0.082, -0.288],
            [-0.072, -0.300],
            [-0.057, -0.318],
        ],
    )
    instep = PchipInterpolator(
        stations,
        [
            [0.075, 0],
            [0.091, -0.058],
            [0.078, -0.155],
            [0.066, -0.201],
            [0.070, -0.224],
            [0.088, -0.242],
            [0.120, -0.260],
            [0.145, -0.277],
            [0.150, -0.297],
        ],
    )
    vertices = []
    faces = []
    panel_indices = []
    # Use matching column counts so both soles share every stitch.
    rings, columns = 28, 16
    u = np.linspace(0, 1, columns + 1)
    arc = 0.5 - 0.5 * np.cos(np.pi * u)
    for panel in range(2):
        sign = 1 if panel == 0 else -1
        indices = np.empty((rings, columns + 1), dtype=np.int32)
        for row, t in enumerate(np.linspace(0, 1, rings)):
            opening = np.exp(-((t / 0.17) ** 2))
            across = (1 - opening) * u + opening * arc
            outline = heel(t) + across[:, None] * (instep(t) - heel(t))
            # A rounded sole and two lightly separated faces leave slack to wrap around a load.
            outline[:, 1] -= 0.044 * np.sin(np.pi * u) * t**3
            profile = np.sin(np.pi * u)
            half_depth = 0.007 + (0.0585 - 0.007) * opening
            depth = half_depth * profile ** (1 - 0.5 * (1 - opening)) * np.clip((1 - t) / 0.11, 0, 1)
            for col in range(columns + 1):
                if panel and (col in (0, columns) or row == rings - 1):
                    indices[row, col] = panel_indices[0][row, col]
                else:
                    indices[row, col] = len(vertices)
                    vertices.append((outline[col, 0], sign * depth[col], outline[col, 1]))
                if row and col:
                    a, b = indices[row - 1, col - 1 : col + 1]
                    c, d = indices[row, col - 1 : col + 1]
                    # At either sole corner, route the diagonal through the panel interior.
                    split_ad = col != 1 and ((row + col) % 2 or (row == rings - 1 and col == columns))
                    pair = [(a, b, d), (a, d, c)] if split_ad else [(a, b, c), (b, d, c)]
                    faces.extend(pair if not panel else [triangle[::-1] for triangle in pair])
        panel_indices.append(indices)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    center = vertices[faces].mean(axis=1)
    colors = np.tile(RED, (len(faces), 1)).astype(np.float32)
    colors[(center[:, 2] > -0.052) | ((center[:, 0] > 0.105) & (center[:, 2] < -0.23))] = CREAM
    cuff = np.r_[panel_indices[0][0, ::-1], panel_indices[1][0, 1:-1]]
    return vertices, faces, colors, cuff


def candy_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a closed twist-wrapper mesh, 64 mm long with a 28 mm candy core.

    The core is aligned along x. Use the complete mesh for cloth contacts and a simple rigid
    core collider for grasping so the fingers cannot snag on decorative folds.
    """
    sides = 24
    x = np.array([-0.032, -0.024, -0.019, -CANDY_CORE_HALF_LENGTH, CANDY_CORE_HALF_LENGTH, 0.019, 0.024, 0.032])
    radius = np.array([0.011, 0.004, 0.006, CANDY_CORE_RADIUS, CANDY_CORE_RADIUS, 0.006, 0.004, 0.011])
    angle = np.arange(sides) * (2 * np.pi / sides)
    vertices = []
    for ring, (position, size) in enumerate(zip(x, radius, strict=True)):
        twist = 0.65 * ring
        folds = 1 + 0.16 * np.cos(6 * angle) if ring in (0, 1, 6, 7) else np.ones(sides)
        vertices.extend(
            zip(
                np.full(sides, position),
                size * folds * np.cos(angle + twist),
                size * folds * np.sin(angle + twist),
                strict=True,
            )
        )
    vertices.extend([(x[0], 0, 0), (x[-1], 0, 0)])
    faces = []
    for ring in range(len(x) - 1):
        for side in range(sides):
            a, b = ring * sides + side, ring * sides + (side + 1) % sides
            faces.extend([(a, b, a + sides), (b, b + sides, a + sides)])
    for side in range(sides):
        nxt = (side + 1) % sides
        faces.extend(
            [(len(x) * sides, nxt, side), (len(x) * sides + 1, (len(x) - 1) * sides + side, (len(x) - 1) * sides + nxt)]
        )
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.array([CREAM if (index // 2) % 6 < 2 else RED for index in range(len(faces))], dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32), faces, colors
