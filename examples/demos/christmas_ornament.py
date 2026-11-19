# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch an Agile One hang a glass ornament on a fir tree, moving only its right arm.

.. code-block:: bash

    export AGILE_ONE_URDF=/path/to/robot_menagerie/agile_robotics/agile_one_sn/generated/hands_wuji/urdf/hands_wuji.urdf
    uv run --extra importers isaaclab demo christmas-ornament

The scene is an :class:`~isaaclab.scene.InteractiveScene` simulated by Newton VBD. The tree is a
USD cable graph: each branch and its twigs are welded ``BasisCurves`` pinned inside a static trunk.
The ornament is a rigid body whose ring collides through a mesh SDF. It starts hanging from the
robot's extended index finger and ends on a branch, held by contact alone: nothing welds,
attaches or kinematically drives it.

The Agile One is internal. ``AGILE_ONE_URDF`` names the ``hands_wuji`` URDF in a
``robot_menagerie`` checkout with its Git LFS meshes, which the demo converts to USD on first
launch; ``AGILE_ONE_USD`` names an already converted USD instead. Pass ``--viz none`` to run
headless.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

FPS = 60
SUBSTEPS = 20
UP = np.array([0.0, 0.0, 1.0])
# Light-object contact: soft enough that the 35 g ornament does not chatter on a 2 mm twig.
RIGID_CONTACT_STIFFNESS = 1.0e4  # [N/m]
RIGID_CONTACT_DAMPING = 30.0  # [N s/m]
# Speculative contact gap [m]. Newton's default is far wider than a 2 mm twig; with it, most of
# the tree's segments register as touching and overflow the per-body contact buffers.
RIGID_GAP = 0.002

# Layout. The tree leaves a slot toward the robot, and its target branch points back at it. The
# robot stands where its right arm alone can hold the finger level along that branch.
HANG_XY = np.array([0.40, -0.10])  # target hang point, horizontally [m]
BRANCH_YAW = math.pi  # the target branch points from the trunk toward the robot
ROBOT_XY = np.array([-0.10, -0.02])  # hang point 0.50 m ahead of and 0.08 m right of the pelvis
HOME = np.array([0.28, -0.24, 0.92])  # hook point at rest, ornament hanging from the finger [m]

# ----------------------------------------------------------------------------------------------
# Tree layout: a static trunk and one welded cable graph per branch
# ----------------------------------------------------------------------------------------------

TREE_HEIGHT = 1.6  # [m]
TRUNK_SEGMENTS = 14
TRUNK_SEGMENT_LENGTH = TREE_HEIGHT / TRUNK_SEGMENTS  # [m]
FIRST_WHORL = 3
TARGET_HEIGHT_RANGE = (0.75, 0.85)  # target whorl root height [m]; its hang point is ~0.93 m up
HANG_FRACTION = 0.5  # hang point along the target's last segment, from its root end
ROBOT_SECTOR = (math.radians(25.0), 0.55, 1.0)  # kept clear: half-angle, min radius [m], max height [m]
CORRIDOR = (0.20, 0.07)  # approach kept clear: length beyond the hang point [m], radius [m]

# [Pa] Branch wood is soft enough that the ornament visibly dips its branch, yet stiff enough that
# the loaded tip stays level. Newton derives each rod joint's bend stiffness from this modulus and
# the curve thickness at the joint (beam theory); twist follows from the Poisson ratio.
BRANCH_YOUNG_MODULUS = 2.56e8
BRANCH_POISSON_RATIO = 1.0 / 3.0  # twist = 0.75 bend for a round section
# Structural stretch and shear stiffness [N]; per joint they are divided by its rest length, about
# 0.09 m on a branch, for 1e5 N/m. Far stiffer leaves bending under-converged in 20 iterations.
BRANCH_STRETCH_STIFFNESS = 9.0e3
# Structural bend damping [N m^2 s]; one value per branch, which damps the thin tips well and the
# thick roots lightly. Bouncing walks a hanging ring outward.
BRANCH_BEND_DAMPING = 2.0 * BRANCH_YOUNG_MODULUS * math.pi * 0.004**4 / 4.0
WOOD_DENSITY = 900.0  # [kg/m^3] needles included
TWIG_DENSITY = 2500.0  # [kg/m^3] thin twig stems carry the needle mass
NEEDLE_RADIUS = 0.0011  # [m]
NEEDLE_HALF_LENGTH = (0.008, 0.011)  # [m]

BARK = (0.36, 0.22, 0.12)
BRANCH_BARK = (0.22, 0.20, 0.10)
TWIG = (0.18, 0.26, 0.10)
BAUBLE_COLORS = ((0.85, 0.1, 0.1), (0.95, 0.75, 0.15), (0.2, 0.35, 0.9), (0.85, 0.85, 0.9))


@dataclass
class Curve:
    """One cable: points, thickness at each point and each segment [m], and density."""

    points: np.ndarray
    point_radii: list[float]
    segment_radii: list[float]
    density: float
    color: tuple[float, float, float]


@dataclass
class Branch:
    """A branch with its twigs. Curve 0 starts with a stub segment pinned inside the trunk."""

    curves: list[Curve]
    junctions: list[tuple[int, int, int, int]]  # (curve, point, curve, point) welded together
    side: np.ndarray  # horizontal unit vector across the branch
    whorl: int


@dataclass
class TreeLayout:
    """Tree geometry in the tree's frame: origin at the trunk base, z up."""

    trunk: list[tuple[np.ndarray, np.ndarray, float]]
    branches: list[Branch]
    target: int  # branch index
    baubles: list[tuple[int, int, np.ndarray, tuple[float, float, float]]]  # branch, point, tip, color
    needle_hosts: list[tuple[np.ndarray, np.ndarray, float, float]]  # segment p0, p1, radius, shade
    hang_point: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _bent_path(start, direction, length, n, droop) -> np.ndarray:
    """Return points along a branch that starts along ``direction`` and arcs downward."""
    d = direction / np.linalg.norm(direction)
    points = [np.array(start, dtype=float)]
    for _ in range(n):
        d = d - droop * UP / n
        d = d / np.linalg.norm(d)
        points.append(points[-1] + (length / n) * d)
    return np.array(points)


def _angle_off(azimuth: float, facing: float) -> float:
    return abs((azimuth - facing + math.pi) % (2.0 * math.pi) - math.pi)


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class _Keepout:
    """The robot's standing sector and the target branch's approach corridor."""

    def __init__(self, facing):
        self.facing = facing
        self.corridor = None

    def blocks(self, points) -> bool:
        half, rmin, zmax = ROBOT_SECTOR
        for p in points:
            if (
                math.hypot(p[0], p[1]) >= rmin
                and p[2] <= zmax
                and _angle_off(math.atan2(p[1], p[0]), self.facing) <= half
            ):
                return True
            if self.corridor is not None and _point_segment_distance(p, *self.corridor) < CORRIDOR[1]:
                return True
        return False


def _branch(root, direction, length, r_base, whorl, keepout, needle_hosts) -> Branch:
    """Lay out one branch chain with up to four twigs per node."""
    n_seg = max(3, round(length / 0.09))
    points = _bent_path(root, direction, length, n_seg, droop=0.15)
    segment_radii = [r_base + (0.004 - r_base) * k / max(n_seg - 1, 1) for k in range(n_seg)]
    # Each joint takes its outer segment's thickness, so a tapering branch stiffens toward the trunk.
    point_radii = [*segment_radii, segment_radii[-1]]
    stub = root - TRUNK_SEGMENT_LENGTH * UP
    curves = [
        Curve(
            np.vstack([stub, points]),
            [r_base, *point_radii],
            [r_base, *segment_radii],
            WOOD_DENSITY,
            BRANCH_BARK,
        )
    ]
    needle_hosts.extend((points[k], points[k + 1], segment_radii[k], 0.8) for k in range(n_seg))
    side = np.cross(direction, UP)
    side /= np.linalg.norm(side)
    junctions = []
    for node in range(n_seg - 1):
        twig_length = 0.5 * length * (1.0 - node / n_seg)
        if twig_length < 0.06:
            continue
        for sign, tilt, spread, scale in (
            (-1, 0.1, 0.7, 1.0),
            (1, 0.1, 0.7, 1.0),
            (-1, -0.35, 0.45, 0.75),
            (1, -0.35, 0.45, 0.75),
        ):
            twig = _bent_path(
                points[node + 1], 0.75 * direction + sign * spread * side + tilt * UP, scale * twig_length, 2, 0.1
            )
            if keepout is not None and keepout.blocks(twig[1:]):
                continue  # pruned where the robot stands or reaches
            # The twig's first point carries the branch's thickness at the weld, which sets the
            # branch joints there; the twig segments keep their own thickness and mass.
            curves.append(Curve(twig, [point_radii[node + 1], 0.002, 0.002], [0.003, 0.002], TWIG_DENSITY, TWIG))
            junctions.append((0, node + 2, len(curves) - 1, 0))
            needle_hosts.extend((twig[j], twig[j + 1], 0.003 - 0.001 * j, 1.0 + 0.25 * j) for j in range(2))
    return Branch(curves, junctions, side, whorl)


def tree_layout(facing_yaw: float = BRANCH_YAW, seed: int = 7) -> TreeLayout:
    """Lay out the fir; its target branch points along ``facing_yaw``.

    Long lower branches are phased around the robot's standing sector, and twigs that would reach
    into it or into the target's approach corridor are pruned.
    """
    rng = np.random.default_rng(seed)
    keepout = _Keepout(facing_yaw)
    needle_hosts = []
    z = np.linspace(0.0, TREE_HEIGHT, TRUNK_SEGMENTS + 1)
    trunk = [
        (z[k] * UP, z[k + 1] * UP, 0.045 + (0.010 - 0.045) * k / (TRUNK_SEGMENTS - 1)) for k in range(TRUNK_SEGMENTS)
    ]
    needle_hosts.extend((p0, p1, r, 1.1) for p0, p1, r in trunk[-3:])
    whorls = list(range(FIRST_WHORL, TRUNK_SEGMENTS - 1))
    target_whorl = next(k for k in whorls if TARGET_HEIGHT_RANGE[0] <= z[k] <= TARGET_HEIGHT_RANGE[1])
    golden = math.pi * (3.0 - math.sqrt(5.0))
    branches, tips = [], []
    target = 0
    hang_point = np.zeros(3)
    # The target whorl comes first, so its approach corridor is known when pruning the rest.
    for k in [target_whorl] + [k for k in whorls if k != target_whorl]:
        frac = (k - FIRST_WHORL) / (TRUNK_SEGMENTS - 2 - FIRST_WHORL)
        length = 0.70 * (1.0 - frac) + 0.14
        n_branches = 6 if frac < 0.6 else 5
        r_base = 0.013 * (1.0 - frac) + 0.005
        elevation = math.radians(10.0 + 25.0 * frac)
        spacing = 2.0 * math.pi / n_branches
        phase = facing_yaw
        if k != target_whorl:
            phase += (whorls.index(k) * golden) % spacing
            if min(_angle_off(phase + i * spacing, facing_yaw) for i in range(n_branches)) < ROBOT_SECTOR[0] + 0.09:
                phase = facing_yaw + 0.5 * spacing
        for i in range(n_branches):
            is_target = k == target_whorl and i == 0
            azimuth = phase + i * spacing + (0.0 if k == target_whorl else rng.uniform(-0.08, 0.08))
            direction = np.array(
                [math.cos(azimuth) * math.cos(elevation), math.sin(azimuth) * math.cos(elevation), math.sin(elevation)]
            )
            branch = _branch(z[k] * UP, direction, length, r_base, k, None if is_target else keepout, needle_hosts)
            branches.append(branch)
            points = branch.curves[0].points
            if is_target:
                target = len(branches) - 1
                hang_point = points[-2] + HANG_FRACTION * (points[-1] - points[-2])
                tangent = (points[-1] - points[-2]) / np.linalg.norm(points[-1] - points[-2])
                keepout.corridor = (hang_point - 0.05 * tangent, hang_point + CORRIDOR[0] * tangent)
            else:
                tips.append((len(branches) - 1, len(points) - 1, points[-1]))

    # Baubles hang from other branch tips, away from the target and the robot.
    candidates = [
        tip
        for tip in tips
        if np.linalg.norm(tip[2] - hang_point) >= 0.25 and not keepout.blocks([tip[2], tip[2] - 0.11 * UP])
    ]
    picks = rng.choice(len(candidates), size=min(14, len(candidates)), replace=False)
    baubles = [(*candidates[index], BAUBLE_COLORS[n % len(BAUBLE_COLORS)]) for n, index in enumerate(picks)]
    return TreeLayout(trunk, branches, target, baubles, needle_hosts, hang_point)


def _quat_z_to(d: np.ndarray) -> np.ndarray:
    """Return the shortest-arc quaternion (x, y, z, w) rotating +Z onto unit ``d``."""
    if d[2] < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([-d[1], d[0], 0.0, 1.0 + d[2]])
    return q / np.linalg.norm(q)


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def needle_mesh(length: float, radius: float, shade: float, rng) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Return cone needles around one segment, in its frame (+Z along it), as one mesh and color."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    cone = 8
    angles = np.linspace(0.0, 2.0 * math.pi, cone, endpoint=False)
    ring = np.stack([NEEDLE_RADIUS * np.cos(angles), NEEDLE_RADIUS * np.sin(angles)], -1)
    index = np.arange(cone)
    template_tris = np.concatenate(
        [
            np.stack([index, (index + 1) % cone, np.full_like(index, cone)], -1),
            np.stack([(index + 1) % cone, index, np.full_like(index, cone + 1)], -1),
        ]
    )
    templates = [
        np.concatenate([np.column_stack([ring, np.full(cone, -h)]), [[0.0, 0.0, h], [0.0, 0.0, -h]]])
        for h in NEEDLE_HALF_LENGTH
    ]
    n_rings = max(1, int(length / 0.004))
    vertices, colors = [], []
    for k in range(n_rings):
        z = -0.5 * length + (k + rng.uniform(0.2, 0.8)) * length / n_rings
        for j in range(5):
            phi = (k * 5 + j) * golden + rng.uniform(-0.3, 0.3)
            theta = math.radians(rng.uniform(40.0, 70.0))
            radial = np.array([math.cos(phi), math.sin(phi), 0.0])
            d = math.sin(theta) * radial + math.cos(theta) * UP
            size = int(rng.integers(0, 2))
            center = np.array([0.0, 0.0, z]) + radius * radial + NEEDLE_HALF_LENGTH[size] * d
            vertices.append(templates[size] @ _rotation_matrix(_quat_z_to(d)).T + center)
            g = shade * rng.uniform(0.75, 1.15)
            colors.append([0.05 * g, 0.38 * g, 0.12 * g])
    triangles = (template_tris[None] + (np.arange(len(vertices)) * (cone + 2))[:, None, None]).reshape(-1, 3)
    color = tuple(float(c) for c in np.clip(np.mean(colors, axis=0), 0.0, 1.0))
    return np.concatenate(vertices).astype(np.float32), triangles.astype(np.int32), color


# ----------------------------------------------------------------------------------------------
# USD authoring: tree and ornament spawners
# ----------------------------------------------------------------------------------------------


def _vec3f(values):
    from pxr import Gf, Vt

    return Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in values])


def _set_color(prim, color) -> None:
    from pxr import UsdGeom

    UsdGeom.Gprim(prim).CreateDisplayColorAttr(_vec3f([color]))


def _contact_material(stage, path: str, friction: float):
    """Author a physics material with the light-object contact gains."""
    from pxr import Sdf, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    prim = material.GetPrim()
    prim.AddAppliedSchema("NewtonMaterialAPI")
    prim.CreateAttribute("newton:contactStiffness", Sdf.ValueTypeNames.Float).Set(RIGID_CONTACT_STIFFNESS)
    prim.CreateAttribute("newton:contactDamping", Sdf.ValueTypeNames.Float).Set(RIGID_CONTACT_DAMPING)
    return material


def _collider(prim, material=None, density: float | None = None) -> None:
    """Make ``prim`` a collider with the tight contact gap, and bind its physics material."""
    from pxr import Sdf, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    prim.AddAppliedSchema("NewtonCollisionAPI")
    prim.CreateAttribute("newton:contactGap", Sdf.ValueTypeNames.Float).Set(RIGID_GAP)
    if density is not None:
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(density)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def _glossy(stage, path: str, color, roughness: float, metallic: float):
    """Author a preview-surface material for glossy baubles."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bauble(stage, path: str, center, radius: float, color, roughness=0.12, metallic=0.85) -> None:
    """Author a glossy sphere that also collides."""
    from pxr import UsdGeom, UsdShade

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.AddTranslateOp().Set(tuple(map(float, center)))
    _set_color(sphere.GetPrim(), color)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(_glossy(stage, f"{path}_look", color, roughness, metallic))


def _attachment(stage, path: str, src0: str, type0: str, index: int, src1: str, coords1, coords0=None) -> None:
    """Author a hard ``PhysicsAttachment`` from a cable site to a point in ``src1``'s frame."""
    from pxr import Sdf

    prim = stage.DefinePrim(path, "PhysicsAttachment")
    prim.CreateRelationship("physics:src0").SetTargets([src0])
    prim.CreateRelationship("physics:src1").SetTargets([src1])
    prim.CreateAttribute("physics:type0", Sdf.ValueTypeNames.Token).Set(type0)
    prim.CreateAttribute("physics:type1", Sdf.ValueTypeNames.Token).Set("xform")
    prim.CreateAttribute("physics:indices0", Sdf.ValueTypeNames.IntArray).Set([index])
    if coords0 is not None:
        prim.CreateAttribute("physics:coords0", Sdf.ValueTypeNames.Vector3fArray).Set(_vec3f([coords0]))
    prim.CreateAttribute("physics:coords1", Sdf.ValueTypeNames.Vector3fArray).Set(_vec3f([coords1]))
    prim.CreateAttribute("physics:attachmentEnabled", Sdf.ValueTypeNames.Bool).Set(True)


def _author_branch(stage, path: str, branch: Branch, trunk_path: str, material) -> list[str]:
    """Author a branch's welded curves and pin its stub segment to the trunk."""
    from pxr import Sdf, UsdGeom, UsdShade

    curve_paths = []
    for index, curve in enumerate(branch.curves):
        curve_path = f"{path}/curve_{index:02d}"
        geom = UsdGeom.BasisCurves.Define(stage, curve_path)
        geom.CreateTypeAttr(UsdGeom.Tokens.linear)
        geom.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        geom.CreatePointsAttr(_vec3f(curve.points))
        geom.CreateCurveVertexCountsAttr([len(curve.points)])
        geom.CreateWidthsAttr([2.0 * r for r in curve.point_radii])
        geom.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        _set_color(geom.GetPrim(), curve.color)
        prim = geom.GetPrim()
        prim.AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
        prim.AddAppliedSchema("PhysicsCollisionAPI")
        prim.CreateAttribute("physics:thicknesses", Sdf.ValueTypeNames.FloatArray).Set(
            [2.0 * r for r in curve.point_radii]
        )
        prim.CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set("point")
        lengths = np.linalg.norm(np.diff(curve.points, axis=0), axis=1)
        # Segment masses match capsules of the segment thickness, so a twig's thicker weld point
        # adds stiffness but no weight.
        masses = [
            curve.density * math.pi * r * r * (length + 4.0 / 3.0 * r)
            for r, length in zip(curve.segment_radii, lengths)
        ]
        prim.CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
        prim.CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set("segment")
        if index == 0:
            # Explicit cross-section frames: the stub's +Y is the branch's side, so the pin sites
            # below are known in the trunk's frame.
            tangents = np.diff(curve.points, axis=0)
            normals = branch.side - (tangents @ branch.side)[:, None] * tangents / np.sum(tangents**2, axis=1)[:, None]
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            geom.CreateNormalsAttr(_vec3f(np.vstack([normals, normals[-1]])))
            geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        curve_paths.append(curve_path)
    for k, (a, point_a, b, point_b) in enumerate(branch.junctions):
        weld = stage.DefinePrim(f"{path}/weld_{k:02d}", "PhysicsAttachment")
        weld.CreateRelationship("physics:src0").SetTargets([curve_paths[a]])
        weld.CreateRelationship("physics:src1").SetTargets([curve_paths[b]])
        weld.CreateAttribute("physics:type0", Sdf.ValueTypeNames.Token).Set("point")
        weld.CreateAttribute("physics:type1", Sdf.ValueTypeNames.Token).Set("point")
        weld.CreateAttribute("physics:indices0", Sdf.ValueTypeNames.IntArray).Set([point_a])
        weld.CreateAttribute("physics:indices1", Sdf.ValueTypeNames.IntArray).Set([point_b])
        weld.CreateAttribute("physics:attachmentEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    # Pin the stub at three non-collinear sites (u along it, s along its +Y), so the branch root
    # joint carries bend and twist like a rod joint to a fixed trunk.
    stub_a, stub_b = branch.curves[0].points[:2]
    for k, (u, s) in enumerate(((0.0, 0.0), (1.0, 0.0), (0.5, 0.02))):
        site = stub_b + u * (stub_a - stub_b) + s * branch.side
        _attachment(stage, f"{path}/pin_{k}", curve_paths[0], "segment", 0, trunk_path, site, coords0=(u, s, 0.0))
    # The stub sits inside the trunk segment below the whorl and touches the one above.
    relationship = stage.GetPrimAtPath(curve_paths[0]).CreateRelationship("physics:filteredPairs")
    for k in (branch.whorl - 1, branch.whorl):
        relationship.AddTarget(f"{trunk_path}/segment_{k:02d}")
    return curve_paths


def spawn_tree(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the fir: a static trunk with the star, branch cable graphs, and hanging baubles."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*translation))
    layout = tree_layout()
    material = _contact_material(stage, f"{prim_path}/contact", friction=0.8)
    wood = _contact_material(stage, f"{prim_path}/wood", friction=0.8)

    trunk_path = f"{prim_path}/trunk"
    UsdGeom.Xform.Define(stage, trunk_path)
    for k, (p0, p1, radius) in enumerate(layout.trunk):
        capsule = UsdGeom.Capsule.Define(stage, f"{trunk_path}/segment_{k:02d}")
        capsule.CreateAxisAttr("Z")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(float(np.linalg.norm(p1 - p0)))
        capsule.AddTranslateOp().Set(tuple(map(float, 0.5 * (p0 + p1))))
        _set_color(capsule.GetPrim(), BARK)
        _collider(capsule.GetPrim(), material)
    top = layout.trunk[-1][1] + 0.035 * UP
    _bauble(stage, f"{trunk_path}/star", top, 0.035, (1.0, 0.8, 0.25), roughness=0.18, metallic=1.0)
    _collider(stage.GetPrimAtPath(f"{trunk_path}/star"), material)

    branch_paths = []
    for index, branch in enumerate(layout.branches):
        branch_paths.append(_author_branch(stage, f"{prim_path}/branch_{index:03d}", branch, trunk_path, wood))
    # Branches of one whorl start inside the same trunk segment.
    by_whorl: dict[int, list[str]] = {}
    for branch, paths in zip(layout.branches, branch_paths, strict=True):
        by_whorl.setdefault(branch.whorl, []).append(paths[0])
    for paths in by_whorl.values():
        for a, path in enumerate(paths):
            relationship = stage.GetPrimAtPath(path).GetRelationship("physics:filteredPairs")
            for other in paths[a + 1 :]:
                relationship.AddTarget(other)

    for n, (branch, point, tip, color) in enumerate(layout.baubles):
        path = f"{prim_path}/bauble_{n:02d}"
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(tuple(map(float, tip - 0.08 * UP)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        _bauble(stage, f"{path}/ball", (0.0, 0.0, 0.0), 0.03, color)
        _collider(stage.GetPrimAtPath(f"{path}/ball"), material, density=300.0)
        string = UsdGeom.Cylinder.Define(stage, f"{path}/string")
        string.CreateRadiusAttr(0.0015)
        string.CreateHeightAttr(0.05)
        string.AddTranslateOp().Set((0.0, 0.0, 0.055))
        _set_color(string.GetPrim(), (0.8, 0.8, 0.8))
        # Newton's viewers hide colliders on a body that also has visual-only shapes, so the
        # string collides too (with nothing but neighbouring needles' twigs).
        _collider(string.GetPrim(), material, density=300.0)
        _attachment(stage, f"{path}/hook", branch_paths[branch][0], "point", point, path, (0.0, 0.0, 0.08))
        body.GetPrim().CreateRelationship("physics:filteredPairs").AddTarget(branch_paths[branch][0])

    # Rod-graph material: one per tree, so every welded graph shares it.
    curve_material = wood.GetPrim()
    for name in ("PhysicsCurvesDeformableMaterialAPI", "NewtonCurvesDeformableMaterialAPI"):
        curve_material.AddAppliedSchema(name)
    for name, value in (
        ("physics:density", WOOD_DENSITY),
        ("physics:youngsModulus", BRANCH_YOUNG_MODULUS),
        ("physics:poissonsRatio", BRANCH_POISSON_RATIO),
        ("physics:curvesStretchStiffness", BRANCH_STRETCH_STIFFNESS),
        ("physics:curvesShearStiffness", BRANCH_STRETCH_STIFFNESS),
        ("newton:curvesBendDamping", BRANCH_BEND_DAMPING),
        ("newton:curvesTwistDamping", 0.75 * BRANCH_BEND_DAMPING),
    ):
        curve_material.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    return root.GetPrim()


RING_INNER_RADIUS = 0.030  # [m] room to thread a twig under the 17 mm Wuji fingertip
RING_TUBE_RADIUS = 0.0025  # [m]
CAP_RADIUS, CAP_HALF_HEIGHT = 0.007, 0.004  # [m]
SPHERE_RADIUS = 0.030  # [m]
ORNAMENT_MASS = 0.035  # [kg]
RING_DENSITY = 1500.0  # [kg/m^3] lacquered wire
CAP_DENSITY = 2700.0  # [kg/m^3] aluminium
RING_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])


def ring_mesh(major_segments: int = 48, minor_segments: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed torus in the local XZ plane (axis +Y), centred on the origin."""
    major = RING_INNER_RADIUS + RING_TUBE_RADIUS
    u = np.linspace(0.0, 2.0 * math.pi, major_segments, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, minor_segments, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radial = major + RING_TUBE_RADIUS * np.cos(vv)
    vertices = np.stack([radial * np.cos(uu), RING_TUBE_RADIUS * np.sin(vv), radial * np.sin(uu)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(major_segments), np.arange(minor_segments), indexing="ij")
    a = i * minor_segments + j
    b = ((i + 1) % major_segments) * minor_segments + j
    c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
    d = i * minor_segments + (j + 1) % minor_segments
    triangles = np.concatenate([np.stack([a, c, b], -1), np.stack([a, d, c], -1)]).reshape(-1, 3)
    return vertices.astype(np.float32), triangles.astype(np.int32)


def spawn_ornament(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the 35 g ornament: an SDF ring at the body origin, with a cap and glass ball below."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        x, y, z, w = orientation
        root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    material = _contact_material(stage, f"{prim_path}/contact", friction=0.5)
    ring_material = _contact_material(stage, f"{prim_path}/ring_contact", friction=0.8)
    gold = (0.85, 0.75, 0.35)

    vertices, triangles = ring_mesh()
    ring = UsdGeom.Mesh.Define(stage, f"{prim_path}/ring")
    ring.CreatePointsAttr(_vec3f(vertices))
    ring.CreateFaceVertexCountsAttr([3] * len(triangles))
    ring.CreateFaceVertexIndicesAttr(triangles.flatten().tolist())
    _set_color(ring.GetPrim(), gold)
    _collider(ring.GetPrim(), ring_material, density=RING_DENSITY)
    UsdPhysics.MeshCollisionAPI.Apply(ring.GetPrim()).CreateApproximationAttr("sdf")
    ring.GetPrim().AddAppliedSchema("NewtonSDFCollisionAPI")
    for name, value in (
        ("newton:sdfMaxResolution", 64),
        ("newton:sdfNarrowBandInner", -0.004),
        ("newton:sdfNarrowBandOuter", 0.004),
        ("newton:sdfPadding", 0.002),
    ):
        kind = Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float
        ring.GetPrim().CreateAttribute(name, kind).Set(value)

    cap_z = -(RING_INNER_RADIUS + 2.0 * RING_TUBE_RADIUS) - CAP_HALF_HEIGHT
    cap = UsdGeom.Cylinder.Define(stage, f"{prim_path}/cap")
    cap.CreateAxisAttr("Z")
    cap.CreateRadiusAttr(CAP_RADIUS)
    cap.CreateHeightAttr(2.0 * CAP_HALF_HEIGHT)
    cap.AddTranslateOp().Set((0.0, 0.0, cap_z))
    _set_color(cap.GetPrim(), gold)
    _collider(cap.GetPrim(), material, density=CAP_DENSITY)

    ring_volume = 2.0 * math.pi * (RING_INNER_RADIUS + RING_TUBE_RADIUS) * math.pi * RING_TUBE_RADIUS**2
    cap_volume = math.pi * CAP_RADIUS**2 * 2.0 * CAP_HALF_HEIGHT
    sphere_volume = 4.0 / 3.0 * math.pi * SPHERE_RADIUS**3
    glass_density = (ORNAMENT_MASS - RING_DENSITY * ring_volume - CAP_DENSITY * cap_volume) / sphere_volume
    _bauble(stage, f"{prim_path}/glass", (0.0, 0.0, cap_z - CAP_HALF_HEIGHT - SPHERE_RADIUS), SPHERE_RADIUS, cfg.color)
    _collider(stage.GetPrimAtPath(f"{prim_path}/glass"), material, density=glass_density)
    return root.GetPrim()


def _spawner_cfgs():
    """Define the spawner configs once Isaac Lab is importable."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class TreeCfg(sim_utils.SpawnerCfg):
        func: object = clone(spawn_tree)

    @configclass
    class OrnamentCfg(sim_utils.SpawnerCfg):
        func: object = clone(spawn_ornament)
        color: tuple[float, float, float] = (0.85, 0.1, 0.1)

    return TreeCfg, OrnamentCfg


# ----------------------------------------------------------------------------------------------
# Robot: Agile One SN with Wuji hands, pelvis welded to the world, right arm only
# ----------------------------------------------------------------------------------------------

PELVIS_HEIGHT = 0.8096  # [m] the asset's initial root height, legs straight
# The ring hangs from the right index finger. Each Wuji phalanx runs along its link's -Z, flexes
# about +X and has its nail toward -Y. With the waist locked, the arm cannot hold the finger
# level nail-up in front of the chest (the forearm would roll past its limit), so the ring rides
# on the thumb side (+X), which faces up when the hand points forward thumb-up, as it rests.
HOOK_LINK = "r_index_finger_distal"
HOOK_OFFSET = wp.vec3(0.0065, 0.0, -0.010)  # thumb side, 10 mm past the last knuckle
# Motion is planned in a hook frame: +X down the finger, +Z out of the surface the ring rests
# on. That is link (-Z, +Y, +X) as (X, Y, Z): this rotation, applied in the link frame.
HOOK_FRAME = wp.quat(0.0, 0.70710678, 0.0, 0.70710678)
FINGER_THICKNESS = 0.016  # [m] Wuji index fingertip across its sides
ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_CURL = 0.8  # fraction of the flexion range for the digits kept out of the ring's way
POSE = {
    **{
        f"r_{finger}_{j}": _CURL * upper
        for finger in ("middle_finger", "ring_finger", "pinky")
        for j, upper in (("mcp_flex", 1.57), ("pip", 2.094), ("dip", 1.57))
    },
    "r_thumb_cmc_flex": _CURL * 1.291,
    "r_thumb_mcp": _CURL * 1.57,
    "r_thumb_ip": _CURL * 1.57,
    # The zero pose holds both forearms forward; the idle left arm hangs at the side instead.
    "left_elbow_joint": 1.4,
}
HAND_LINKS = ("r_", "right_wrist_yaw_link")  # only the right hand collides
MAX_WAYPOINT_JUMP = 0.5  # [rad] a larger arm-joint step between waypoints is an IK branch switch


def robot_usd(args) -> str:
    """Return the Agile One USD, converting the URDF once if only that is given."""
    if args.robot_usd:
        return args.robot_usd
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # Merging fixed joints drops the Wuji hand's mount offset in this importer, so keep them.
    cfg = UrdfConverterCfg(asset_path=args.robot_urdf, fix_base=True, merge_fixed_joints=False, make_instanceable=False)
    return UrdfConverter(cfg).usd_path


def robot_cfg(usd_path: str, arm_pose: dict[str, float]):
    """Return the Agile One with its pelvis fixed and PD-held joints."""
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path, articulation_props=NewtonArticulationCfg(self_collision_enabled=False)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(float(ROBOT_XY[0]), float(ROBOT_XY[1]), PELVIS_HEIGHT),
            joint_pos={**POSE, **arm_pose},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(joint_names_expr=[".*_(hip|knee|ankle)_.*"], stiffness=5.0e4, damping=500.0),
            "fingers": ImplicitActuatorCfg(joint_names_expr=["(l|r)_.*"], stiffness=1500.0, damping=8.0),
            "body": ImplicitActuatorCfg(
                joint_names_expr=["(waist|head|left_(shoulder|elbow|wrist)|right_(shoulder|elbow|wrist))_.*"],
                stiffness=3.2e4,
                damping=240.0,
            ),
        },
    )


def _short(label: str) -> str:
    return label.split("/")[-1]


def hook_rotation(body_rotation) -> np.ndarray:
    """Return the world hook-frame orientation (x, y, z, w) from the hook link's."""
    return np.array(wp.mul(wp.quat(*body_rotation), HOOK_FRAME))


class ArmIK:
    """Hook-point IK for the right arm on a reference Newton model of the robot USD."""

    def __init__(self, usd_path: str, device):
        import newton

        builder = newton.ModelBuilder()
        pelvis = wp.transform(wp.vec3(float(ROBOT_XY[0]), float(ROBOT_XY[1]), PELVIS_HEIGHT), wp.quat_identity())
        builder.add_usd(usd_path, xform=pelvis)
        for joint, label in enumerate(builder.joint_label):
            if _short(label) in POSE:
                builder.joint_q[builder.joint_q_start[joint]] = POSE[_short(label)]
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.hook = next(i for i, label in enumerate(self.model.body_label) if _short(label) == HOOK_LINK)
        starts = self.model.joint_q_start.numpy()
        self.coord = {_short(label): int(starts[j]) for j, label in enumerate(self.model.joint_label)}
        self.arm = np.array([self.coord[name] for name in ARM_JOINTS])
        self.rest = self.model.joint_q.numpy().copy()
        self.fixed = np.ones(self.rest.shape, dtype=bool)
        self.fixed[self.arm] = False
        self.lower = self.model.joint_limit_lower.numpy()[self.arm]
        self.upper = self.model.joint_limit_upper.numpy()[self.arm]

    def hook_pose(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the hook point and hook-frame orientation for joint coordinates ``coords``."""
        import newton

        newton.eval_fk(
            self.model, wp.array(coords, dtype=wp.float32, device=self.model.device), self.model.joint_qd, self.state
        )
        body = self.state.body_q.numpy()[self.hook]
        return np.array(wp.transform_point(wp.transform(*body), HOOK_OFFSET)), hook_rotation(body[3:])

    def error(self, coords, target, quat) -> tuple[float, float]:
        point, rotation = self.hook_pose(coords)
        return float(np.linalg.norm(point - target)), 2.0 * math.acos(min(1.0, abs(float(np.dot(rotation, quat)))))

    def _solver(self, problems: int, target, quat):
        import newton.ik as ik

        device = self.model.device
        position = ik.IKObjectivePosition(
            link_index=self.hook,
            link_offset=HOOK_OFFSET,
            target_positions=wp.array([wp.vec3(*map(float, target))] * problems, dtype=wp.vec3, device=device),
        )
        rotation = ik.IKObjectiveRotation(
            link_index=self.hook,
            link_offset_rotation=HOOK_FRAME,
            target_rotations=wp.array([wp.vec4(*map(float, quat))] * problems, dtype=wp.vec4, device=device),
        )
        limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower, joint_limit_upper=self.model.joint_limit_upper, weight=10.0
        )
        solver = ik.IKSolver(
            model=self.model,
            n_problems=problems,
            objectives=[position, rotation, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        return solver, position, rotation

    def multistart(self, target, quat, seeds: int = 128) -> list[np.ndarray]:
        """Return the solutions converged to 1 mm and 0.5 degrees from random right-arm seeds."""
        solver, _, _ = self._solver(seeds, target, quat)
        coords = np.tile(self.rest, (seeds, 1))
        coords[:, self.arm] = np.random.default_rng(0).uniform(self.lower, self.upper, size=(seeds, len(self.arm)))
        q = wp.array(coords.astype(np.float32), dtype=wp.float32, device=self.model.device)
        for _ in range(60):
            solver.step(q, q, iterations=10)
            values = q.numpy()
            values[:, self.fixed] = self.rest[self.fixed]
            q.assign(values)
        converged = []
        for candidate in q.numpy():
            error, angle = self.error(candidate, target, quat)
            if error < 0.001 and angle < math.radians(0.5):
                converged.append(candidate.copy())
        return converged

    def solve_rest(self, target, quat) -> np.ndarray:
        """Solve the rest pose, keeping the arm configuration farthest from its joint limits.

        The arm reaches most hook poses with several elbow and wrist configurations, and only
        some continue smoothly along the motion.
        """
        solutions = self.multistart(target, quat)
        if not solutions:
            raise RuntimeError(f"Unreachable rest pose {np.round(target, 3).tolist()}")
        margin = lambda c: float(np.min(np.minimum(c[self.arm] - self.lower, self.upper - c[self.arm])))  # noqa: E731
        self.rest = max(solutions, key=margin)
        return self.solve_path(np.asarray(target)[None], np.asarray(quat)[None])[0]

    def solve_path(self, targets: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        """Solve one joint-coordinate vector per hook waypoint, starting from the rest pose.

        Each waypoint starts from the previous solution; where that stalls, IK re-solves from
        random seeds and keeps the solution nearest the previous one. Raises if a waypoint misses
        by more than 3 mm or needs an arm-joint jump above ``MAX_WAYPOINT_JUMP``.
        """
        solver, position, rotation = self._solver(1, targets[0], rotations[0])
        q = wp.array(self.rest.reshape(1, -1), dtype=wp.float32, device=self.model.device)
        previous = self.rest
        solutions = []
        for target, quat in zip(targets, rotations, strict=True):
            position.set_target_position(0, wp.vec3(*map(float, target)))
            rotation.set_target_rotation(0, wp.vec4(*map(float, quat)))
            # 20 rounds converge most waypoints; awkward wrist poses take up to 80.
            for round_ in range(80):
                solver.step(q, q, iterations=10)
                values = q.numpy().ravel()
                values[self.fixed] = self.rest[self.fixed]
                q.assign(values.reshape(1, -1))
                if round_ >= 19 and round_ % 10 == 9:
                    error, _ = self.error(values, target, quat)
                    if error < 0.001:
                        break
            coords = q.numpy().ravel().copy()
            if error > 0.001:
                candidates = self.multistart(target, quat)
                if candidates:
                    coords = min(candidates, key=lambda c: float(np.max(np.abs(c[self.arm] - previous[self.arm]))))
                    error, _ = self.error(coords, target, quat)
                    q.assign(coords.reshape(1, -1))
            if error > 0.003:
                raise RuntimeError(f"Unreachable hook waypoint {np.round(target, 3).tolist()}: IK error {error:.4f} m")
            jump = float(np.max(np.abs(coords[self.arm] - previous[self.arm])))
            if solutions and jump > MAX_WAYPOINT_JUMP:
                raise RuntimeError(
                    f"IK branch switch at {np.round(target, 3).tolist()}: arm joints jump {jump:.2f} rad"
                )
            solutions.append(coords)
            previous = coords
        return np.array(solutions, dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Motion: hook-point waypoints in the target branch's frame
# ----------------------------------------------------------------------------------------------

PHASES = (
    ("settle", 2.0),
    ("present", 2.5),
    ("align", 1.5),
    ("thread", 2.0),
    ("lower", 2.0),
    ("release", 1.0),
    ("withdraw", 2.0),
    ("retreat", 1.5),
    ("home", 2.0),
    ("admire", 5.0),
)
# Outer branches slope up toward their tips, and a finger copying that slope sheds the ring, so
# the finger aims along the branch in plan view but stays pitched tip-up.
FINGER_PITCH = math.radians(5.0)
SAMPLES_PER_PHASE = 8


def finger_rotation(tangent: np.ndarray) -> np.ndarray:
    """Return the hook-frame orientation (x, y, z, w): +X along -``tangent`` in plan view,
    pitched tip-up, and +Z as close to up as possible."""
    inward = -np.asarray(tangent[:2], dtype=float)
    inward /= np.linalg.norm(inward)
    x_axis = np.array([inward[0] * math.cos(FINGER_PITCH), inward[1] * math.cos(FINGER_PITCH), math.sin(FINGER_PITCH)])
    z_axis = UP - np.dot(UP, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    m = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    return np.array(wp.quat_from_matrix(wp.mat33(*m.flatten())))


def choreography(hang, tangent, ring_top_offset: float, twig_radius: float):
    """Return each phase's end time and hook-point position, and the hook orientation.

    The ornament starts hanging from the finger, its ring's inner top ``ring_top_offset`` above
    the hook point. The twig threads through the ring beside and under the finger; the finger then
    comes straight down past the twig, so the ring lands on it and the finger ends at the centre
    of the ring's hole, touching neither; it then slides back out along its own axis. The finger
    stays level, which the right arm alone can hold in front of the robot.
    """
    tangent = tangent / np.linalg.norm(tangent)
    carry = finger_rotation(tangent)
    finger_dir = np.array(wp.quat_rotate(wp.quat(*carry), wp.vec3(1.0, 0.0, 0.0)))
    lateral = twig_radius + 0.5 * FINGER_THICKNESS + 0.004  # 4 mm beside the twig
    if RING_INNER_RADIUS - lateral - 0.5 * FINGER_THICKNESS < 0.005:
        raise ValueError("The ring is too small to lower the finger beside the twig.")
    side = np.cross(UP, tangent)
    side /= np.linalg.norm(side)
    over = hang + lateral * side
    thread = over + 0.5 * (FINGER_THICKNESS + 2.0 * RING_INNER_RADIUS - ring_top_offset) * UP
    align = thread + 0.10 * tangent
    present = align + 0.05 * UP
    # Down until the finger's centre is where the ring's centre will be once it hangs on the twig.
    ring_centre = hang[2] + twig_radius + ring_top_offset - RING_INNER_RADIUS
    lower = np.array([over[0], over[1], ring_centre + 0.5 * FINGER_THICKNESS])
    withdraw = lower - 0.15 * finger_dir
    retreat = withdraw - 0.05 * UP
    positions = np.array([HOME, present, align, thread, lower, lower, withdraw, retreat, HOME, HOME])
    return np.cumsum([duration for _, duration in PHASES]), positions, carry


def densify(times, positions):
    """Expand each phase into poses on its straight segment at minimum-jerk progress.

    Interpolating joints between whole-phase IK solutions swung the hand off its path, pressed
    the twig down and catapulted the ornament; IK on these dense poses keeps the hand on it.
    """
    dense_t, dense_p = [0.0], [HOME]
    t0, p0 = 0.0, HOME
    for t1, p1 in zip(times, positions, strict=True):
        for k in range(1, SAMPLES_PER_PHASE + 1):
            u = k / SAMPLES_PER_PHASE
            dense_t.append(t0 + u * (t1 - t0))
            dense_p.append(p0 + u**3 * (10.0 - 15.0 * u + 6.0 * u**2) * (p1 - p0))
        t0, p0 = t1, p1
    return np.array(dense_t), np.array(dense_p)


def sample_linear(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate the rows of ``targets`` piecewise-linearly at ``time``."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    return targets[segment] + alpha * (targets[segment + 1] - targets[segment])


# ----------------------------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------------------------


def tree_origin(layout: TreeLayout) -> np.ndarray:
    """Place the tree so its target branch's hang point lands on ``HANG_XY``."""
    return np.array([*(HANG_XY - layout.hang_point[:2]), 0.0])


def scene_cfg(usd_path: str, arm_pose: dict[str, float], ornament_pose: tuple, layout: TreeLayout):
    """Return the scene: ground, light, robot, ornament and tree."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.cloner import CloneCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    TreeCfg, OrnamentCfg = _spawner_cfgs()
    position, rotation = ornament_pose

    @configclass
    class OrnamentSceneCfg(InteractiveSceneCfg):
        clone_cfg = CloneCfg(clone_template="/World/Env_{}")
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        robot = robot_cfg(usd_path, arm_pose)
        ornament = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ornament",
            spawn=OrnamentCfg(),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(map(float, position)), rot=tuple(map(float, rotation))),
        )
        tree = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tree",
            spawn=TreeCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(map(float, tree_origin(layout)))),
        )

    return OrnamentSceneCfg(num_envs=1, env_spacing=0.0)


def _prepare_world(layout: TreeLayout, joint_pose: dict[str, float]):
    """Return a Newton world hook: rest pose, hand-only robot collision, and branch needles.

    The robot starts in its rest pose so that the ornament, spawned on the finger, starts at rest
    too. Cable segments have no USD prims, so their needles are visual-only shapes added to the
    segment bodies, which the Newton visualizers draw.
    """
    import newton

    origin = tree_origin(layout)
    trunk_hosts = layout.needle_hosts[:3]
    hosts = [(origin + 0.5 * (p0 + p1), p0, p1, r, shade) for p0, p1, r, shade in layout.needle_hosts[3:]]

    def hook(builder, world, position, rotation) -> None:
        for joint, label in enumerate(builder.joint_label):
            if "/Robot/" in label and _short(label) in joint_pose:
                builder.joint_q[builder.joint_q_start[joint]] = joint_pose[_short(label)]
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for shape, body in enumerate(builder.shape_body):
            label = builder.body_label[body] if body >= 0 else ""
            if "/Robot/" in label and not _short(label).startswith(HAND_LINKS):
                builder.shape_flags[shape] &= ~collide
        # Cables import without their display color: thin twigs green-brown, branches bark.
        for shape, label in enumerate(builder.shape_label):
            if "/Tree/branch_" in label:
                builder.shape_color[shape] = TWIG if builder.shape_scale[shape][0] <= 0.003 else BRANCH_BARK
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        rng = np.random.default_rng(11)
        for p0, p1, radius, shade in trunk_hosts:
            vertices, triangles, color = needle_mesh(float(np.linalg.norm(p1 - p0)), radius, shade, rng)
            mesh = newton.Mesh(vertices, triangles.flatten(), compute_inertia=False)
            center = np.asarray(position) + origin + 0.5 * (p0 + p1)
            builder.add_shape_mesh(
                -1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), mesh=mesh, cfg=visual, color=color
            )
        tree_bodies = [b for b, label in enumerate(builder.body_label) if "/Tree/" in label]
        body_centers = np.array([list(builder.body_q[b])[:3] for b in tree_bodies]) - np.asarray(position)
        for center, p0, p1, radius, shade in hosts:
            distance = np.linalg.norm(body_centers - center, axis=1)
            if distance.min() > 1.0e-4:
                raise RuntimeError(f"No cable segment at {np.round(center, 3).tolist()} for its needles.")
            vertices, triangles, color = needle_mesh(float(np.linalg.norm(p1 - p0)), radius, shade, rng)
            mesh = newton.Mesh(vertices, triangles.flatten(), compute_inertia=False)
            builder.add_shape_mesh(tree_bodies[int(distance.argmin())], mesh=mesh, cfg=visual, color=color)

    return hook


def simulation_cfg(device: str | None):
    """Return the VBD simulation and Newton GL view of the scene."""
    import newton
    from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg, VBDSolver, VBDSolverCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass
    from isaaclab.visualizers import VisualizerCfg

    class RestPoseVBDSolver(VBDSolver):
        """Place the bodies at the authored joint coordinates before VBD records its rest state."""

        def build(self, model, mode, env_template):
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)
            return super().build(model, mode, env_template)

    @configclass
    class RestPoseVBDSolverCfg(VBDSolverCfg):
        class_type: type = RestPoseVBDSolver

    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=device,
        physics=NewtonCfg(
            solver_cfg=RestPoseVBDSolverCfg(
                iterations=20,
                update_joint_state=False,
                rigid_joint_linear_ke=1.0e7,
                rigid_joint_angular_ke=1.0e6,
                rigid_body_contact_buffer_size=256,
                friction_epsilon=1.0e-3,
            ),
            num_substeps=SUBSTEPS,
            # Collide every substep: the ring rides on millimetre clearances.
            collision_decimation=1,
            default_shape_cfg=NewtonShapeCfg(
                gap=RIGID_GAP, ke=RIGID_CONTACT_STIFFNESS, kd=RIGID_CONTACT_DAMPING, mu=0.8
            ),
        ),
        visualizer_cfgs=VisualizerCfg(
            eye=(-0.9, -2.3, 1.25),
            lookat=(0.55, 0.0, 0.80),
            enable_live_plots=False,
            backend=NewtonGLBackendCfg(show_particles=False),
        ),
    )


class OrnamentHanging:
    """Plan the hand-over, build the scene and run the scripted arm through Isaac Lab.

    The rest pose and the scene config are solved before launch, so the launcher sees every
    camera the scene needs; :meth:`start` then builds the scene.
    """

    def __init__(self, usd_path: str, device: str):
        self.sim_time = 0.0
        self.duration = sum(duration for _, duration in PHASES)
        self.carry = finger_rotation(np.array([math.cos(BRANCH_YAW), math.sin(BRANCH_YAW), 0.0]))
        self.layout = tree_layout()

        self.ik = ArmIK(usd_path, device)
        self.q_home = self.ik.solve_rest(HOME, self.carry)
        hook_point, hook_quat = self.ik.hook_pose(self.q_home)
        # The ornament hangs from the finger: ring inner top on the hook point, ring axis along it.
        back = np.array(wp.quat_rotate(wp.quat(*hook_quat), wp.vec3(0.0, 0.0, 1.0)))
        rotation = wp.mul(wp.quat(*hook_quat), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * math.pi))
        ring_center = hook_point - (RING_INNER_RADIUS - 0.0005) * back
        self.arm_pose = {name: float(self.q_home[self.ik.coord[name]]) for name in ARM_JOINTS}
        self.scene_cfg = scene_cfg(usd_path, self.arm_pose, (ring_center, tuple(rotation)), self.layout)

    def start(self, sim) -> None:
        """Build the scene and find the bodies the motion and its planning follow."""
        from isaaclab_newton.cloner import newton_builder_world_hook

        from isaaclab.scene import InteractiveScene

        self.sim = sim
        with newton_builder_world_hook(_prepare_world(self.layout, {**POSE, **self.arm_pose})):
            self.scene = InteractiveScene(self.scene_cfg)
            sim.reset()
        self.robot = self.scene["robot"]
        self.arm_ids = [self.robot.joint_names.index(name) for name in ARM_JOINTS]
        # Every joint holds its rest pose; the arm's targets follow the motion from here on.
        self.targets = self.robot.data.default_joint_pos.torch.clone()
        backend = sim.backend_manager.backend
        self.backend = backend
        labels = list(backend.model.body_label)
        self.ornament = next(i for i, label in enumerate(labels) if label.endswith("/Ornament"))
        self.hook_body = next(i for i, label in enumerate(labels) if _short(label) == HOOK_LINK)
        # The hang segment is the target branch's outermost cable segment.
        points = self.layout.branches[self.layout.target].curves[0].points + tree_origin(self.layout)
        body_q = backend.state_0.body_q.numpy()
        center = 0.5 * (points[-2] + points[-1])
        tree = [i for i, label in enumerate(labels) if "/Tree/" in label]
        self.hang_body = tree[int(np.argmin(np.linalg.norm(body_q[tree, :3] - center, axis=1)))]
        self.hang_local = np.array([0.0, 0.0, (HANG_FRACTION - 0.5) * np.linalg.norm(points[-1] - points[-2])])
        self.twig_radius = self.layout.branches[self.layout.target].curves[0].segment_radii[-1]
        # Hold the rest pose through the settle; the rest is planned from the settled branch.
        self.times = np.array([0.0, PHASES[0][1]])
        self.q_targets = np.stack([self.q_home, self.q_home])
        self.planned = False

    def hang_frame(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the target's hang point and unit tangent toward its tip."""
        pose = wp.transform(*body_q[self.hang_body])
        tangent = np.array(wp.quat_rotate(pose.q, wp.vec3(0.0, 0.0, 1.0)))
        return np.array(wp.transform_point(pose, wp.vec3(*self.hang_local))), tangent

    def plan(self) -> None:
        """Solve the hand-over from the settled branch and the ring as it rides on the finger."""
        body_q = self.backend.state_0.body_q.numpy()
        hang, tangent = self.hang_frame(body_q)
        hook = np.array(wp.transform_point(wp.transform(*body_q[self.hook_body]), HOOK_OFFSET))
        back = np.array(wp.quat_rotate(wp.quat(*hook_rotation(body_q[self.hook_body, 3:])), wp.vec3(0.0, 0.0, 1.0)))
        ring_top = body_q[self.ornament, :3] + RING_INNER_RADIUS * back
        times, positions, carry = choreography(hang, tangent, float(np.dot(ring_top - hook, back)), self.twig_radius)
        dense_t, dense_p = densify(times, positions)
        q = self.ik.solve_path(dense_p[1:], np.tile(carry, (len(dense_p) - 1, 1)))
        self.times, self.q_targets = dense_t, np.vstack([self.q_home, q])
        self.planned = True

    def step(self) -> None:
        """Apply one frame of scripted arm targets and advance the physics."""
        if not self.planned and self.sim_time >= PHASES[0][1] - 1e-9:
            self.plan()
        arm = sample_linear(self.sim_time, self.times, self.q_targets)[self.ik.arm]
        self.targets[0, self.arm_ids] = torch.tensor(arm, dtype=torch.float32, device=self.targets.device)
        self.robot.actuators.target_command.set_position_index(value=self.targets)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(1.0 / FPS)
        self.sim_time += 1.0 / FPS


def main() -> None:
    """Launch the scene and run one complete hand-over."""
    parser = argparse.ArgumentParser(
        description="Hang a glass ornament on a fir tree with a scripted Agile One in Isaac Lab.",
        epilog="Set AGILE_ONE_URDF or AGILE_ONE_USD to the Agile One hands_wuji asset. Use --viz none for headless.",
    )
    parser.add_argument("--robot_urdf", default=os.environ.get("AGILE_ONE_URDF"), help="Agile One hands_wuji URDF.")
    parser.add_argument(
        "--robot_usd", default=os.environ.get("AGILE_ONE_USD"), help="Converted Agile One USD; skips conversion."
    )
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Limit the run to this many frames. Zero runs it in full."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if not (args.robot_usd and os.path.isfile(args.robot_usd)) and not (
        args.robot_urdf and os.path.isfile(args.robot_urdf)
    ):
        parser.error("Set AGILE_ONE_URDF (or AGILE_ONE_USD) to the Agile One hands_wuji asset.")
    if args.max_steps < 0:
        parser.error("--max_steps must be nonnegative.")

    import isaaclab.sim as sim_utils

    # Convert before launching: the importer needs a fresh USD session.
    usd_path = robot_usd(args)
    sim_cfg = simulation_cfg(args.device)
    demo = OrnamentHanging(usd_path, sim_cfg.device)
    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=demo.scene_cfg), args):
        sim = sim_utils.SimulationContext(sim_cfg)
        demo.start(sim)
        frames = math.ceil(demo.duration * FPS)
        if args.max_steps:
            frames = min(frames, args.max_steps)
        for _ in range(frames):
            if not sim.is_running():
                break
            demo.step()


if __name__ == "__main__":
    main()
