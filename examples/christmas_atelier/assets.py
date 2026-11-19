# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible, meter-scale stocking and wrapped-candy meshes; no external art tools required."""

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

RED = (0.68, 0.025, 0.045)
CREAM = (0.96, 0.90, 0.75)
GREEN = (0.025, 0.30, 0.13)
GOLD = (0.95, 0.60, 0.08)
CANDY_CORE_RADIUS = 0.014
CANDY_CORE_HALF_LENGTH = 0.014
CANDY_CAPSULE_HALF_LENGTH = 0.005


def stocking_mesh(rings: int = 28, sides: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices, triangles, face colors and supported mouth indices.

    The single continuous shell bends from a closed toe into an upright leg. Local origin is
    the center of the open cuff; z points upward. Shared vertices join all panels and the toe.
    Below the supported cuff, the rest cross-section is flattened like an empty garment,
    preserving the fabric circumference rather than prescribing an inflated tube.
    """
    if rings < 8 or sides < 8:
        raise ValueError("The stocking needs at least eight rings and eight sides.")
    # Arc-length resampling avoids skinny triangles at the heel and cuff.
    control = np.array([[0.15, -0.31], [0.10, -0.31], [0.035, -0.29], [0.0, -0.235], [0.0, 0.0]])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(control, axis=0), axis=1))]
    samples = np.linspace(0, distance[-1], rings)
    centers = np.column_stack([np.interp(samples, distance, control[:, axis]) for axis in range(2)])
    tangent = np.gradient(centers, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    radius = np.interp(
        samples / samples[-1], [0, 0.10, 0.30, 0.48, 0.90, 1], [0.018, 0.043, 0.053, 0.063, 0.073, 0.075]
    )
    angle = np.arange(sides) * (2 * np.pi / sides)
    # A relaxed sock is flattened below its supported mouth. Preserve each ring's fabric
    # circumference while changing its rest cross-section, rather than shrinking the garment.
    depth = np.interp(samples / samples[-1], [0, 0.78, 0.94, 1], [0.35, 0.35, 0.55, 0.78])
    sections = np.stack([np.broadcast_to(np.cos(angle), (rings, sides)), depth[:, None] * np.sin(angle)], axis=-1)
    circumference = np.linalg.norm(np.roll(sections, -1, axis=1) - sections, axis=-1).sum(axis=1)
    mouth_section = np.column_stack([np.cos(angle), 0.78 * np.sin(angle)])
    mouth_circumference = np.linalg.norm(np.roll(mouth_section, -1, axis=0) - mouth_section, axis=-1).sum()
    sections *= (mouth_circumference / circumference)[:, None, None]
    vertices = np.zeros((rings, sides, 3))
    vertices[:, :, 0] = centers[:, 0, None] + normal[:, 0, None] * radius[:, None] * sections[:, :, 0]
    vertices[:, :, 1] = radius[:, None] * sections[:, :, 1]
    vertices[:, :, 2] = centers[:, 1, None] + normal[:, 1, None] * radius[:, None] * sections[:, :, 0]
    vertices = np.vstack([vertices.reshape(-1, 3), [*centers[0, :1], 0.0, centers[0, 1]]])
    faces = []
    for ring in range(rings - 1):
        for side in range(sides):
            a, b = ring * sides + side, ring * sides + (side + 1) % sides
            faces.extend([(a, b, a + sides), (b, b + sides, a + sides)])
    faces.extend([(rings * sides, (side + 1) % sides, side) for side in range(sides)])
    faces = np.asarray(faces, dtype=np.int32)
    midpoint = vertices[faces].mean(axis=1)
    colors = np.tile(RED, (len(faces), 1))
    colors[(midpoint[:, 2] > -0.045) | (midpoint[:, 0] > 0.125)] = CREAM
    cuff = np.arange((rings - 1) * sides, rings * sides, dtype=np.int32)
    return vertices.astype(np.float32), faces, colors.astype(np.float32), cuff


def panel_stocking_mesh(rings: int = 28, sides: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shape two cloth panels around an open cuff and join their side and sole seams.

    Rows run across the garment. The two side outlines independently shape the heel and
    instep. The curved sole closes on a line of shared vertices, as on a sewn stocking.
    """
    if rings < 8 or sides < 8:
        raise ValueError("The stocking needs at least eight rows and eight sides.")

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
    columns = (sides + 1) // 2
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
    # Counterclockwise from above for the temporary containment cap.
    cuff = np.r_[panel_indices[0][0, ::-1], panel_indices[1][0, 1:-1]]
    return vertices, faces, colors, cuff


def candy_mesh(sides: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a closed twist-wrapper mesh, 64 mm long with a 28 mm candy core.

    The core is aligned along x. Use the complete mesh for cloth contacts and a simple rigid
    core collider for grasping so the fingers cannot snag on decorative folds.
    """
    if sides < 8:
        raise ValueError("The candy needs at least eight sides.")
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


def export_assets(directory: Path, *, sewn: bool = False) -> None:
    """Write portable USD art meshes with per-face colors and no backend-specific physics."""
    from pxr import Gf, Usd, UsdGeom, Vt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stocking = panel_stocking_mesh() if sewn else stocking_mesh()
    for name, geometry in (("stocking", stocking), ("candy", candy_mesh())):
        vertices, faces, colors = geometry[:3]
        stage = Usd.Stage.CreateNew(str(directory / f"{name}.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        mesh = UsdGeom.Mesh.Define(stage, f"/{name.title()}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray.FromNumpy(colors))
        if name == "stocking":
            cuff = UsdGeom.Subset.Define(stage, "/Stocking/SupportedCuff")
            cuff.CreateElementTypeAttr(UsdGeom.Tokens.point)
            cuff.CreateIndicesAttr(geometry[3].tolist())
        mesh.CreateExtentAttr(
            [Gf.Vec3f(*map(float, vertices.min(axis=0))), Gf.Vec3f(*map(float, vertices.max(axis=0)))]
        )
        stage.SetDefaultPrim(mesh.GetPrim())
        stage.GetRootLayer().Save()
