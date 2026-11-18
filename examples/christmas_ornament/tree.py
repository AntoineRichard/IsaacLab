# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A conifer built from VBD rod joints, with render-only needles.

Every rod segment is a capsule body; neighbours are joined with ``add_joint_rod``. Bend
stiffness comes from beam theory, ``E * (pi r^4 / 4) / span``, with two moduli: a stiff trunk
that stands, and young-wood branches that visibly dip under an ornament. Twig capsules are thin
stems whose density includes the needles drawn on them; needles never enter the simulation.

The tree leaves a slot in front of it, toward ``facing_yaw``: long lower branches are phased
around it and twigs that would reach into it are pruned, so a robot can stand there and reach
one outer *target branch* along an unobstructed corridor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import newton
import numpy as np
import warp as wp

TRUNK_YOUNG_MODULUS = 5.0e9  # [Pa] green softwood; the trunk must stand
# [Pa] Branch and twig wood, softer than green softwood (~5-10 GPa) so an ornament visibly dips
# a branch (~5 mm), yet stiff enough that the loaded tip stays level: much softer and the last
# twig joint tips downhill under the ring, which then creeps off the end.
BRANCH_YOUNG_MODULUS = 2.0e8
TWIST_RATIO = 0.75  # GJ / EI for a round section (G ~ E / 2.6, J = 2 I)
BEND_DAMPING_RATIO = 0.15  # [s] bend damping / bend stiffness; needle-laden branches are well damped
# Rod stretch stiffness [N/m]. The trunk carries the whole tree; branches carry < 2 N. Keeping
# branch stretch near the bend scale matters: at 1e7 N/m against bend stiffness of 2-50 N m/rad,
# 20 VBD iterations leave branch bending ~10x under-converged and branches sag from solver
# softness rather than material.
TRUNK_STRETCH_STIFFNESS = 1.0e7
BRANCH_STRETCH_STIFFNESS = 1.0e5
WOOD_DENSITY = 900.0  # [kg/m^3] trunk and branches, needles included
TWIG_DENSITY = 2500.0  # [kg/m^3] thin twig stems carry the needle mass
HANGER_BEND = 2.0e-3  # [N m/rad] ornament strings barely resist bending
NEEDLE_RADIUS = 0.0011  # [m]
NEEDLE_HALF_LENGTH = (0.008, 0.011)  # [m] two size classes, one instanced batch each

HEIGHT = 1.6
TRUNK_SEGMENTS = 14
FIRST_WHORL = 3
TARGET_HEIGHT_RANGE = (0.95, 1.15)  # target whorl root height above the tree origin [m]
HANG_FRACTION = 0.5  # hang point along the target's last segment, from its root end
ROBOT_SECTOR = (math.radians(25.0), 0.55, 1.0)  # half-angle, min radius [m], max height [m]
CORRIDOR = (0.20, 0.07)  # length outward from the hang point [m], clearance radius [m]

BARK = (0.36, 0.22, 0.12)
BRANCH_BARK = (0.22, 0.20, 0.10)
TWIG = (0.18, 0.26, 0.10)
ORNAMENT_COLORS = ((0.85, 0.1, 0.1), (0.95, 0.75, 0.15), (0.2, 0.35, 0.9), (0.85, 0.85, 0.9))


@dataclass
class TargetBranch:
    """The outer branch the robot decorates."""

    segment_bodies: list[int]
    joints: list[int]
    tip_body: int
    hang_body: int
    hang_local: np.ndarray  # (3,) point on the hang segment's axis, body frame [m]
    tangent_local: np.ndarray  # (3,) unit +Z of the hang segment, pointing toward the tip
    twig_radius: float  # wood radius of the hang segment [m]
    hang_neighbours: list[int] = field(default_factory=list)  # fork twigs next to the hang point


@dataclass
class Tree:
    bodies: list[int]
    joints: list[int]
    trunk_joints: list[int]
    trunk_top_body: int
    target: TargetBranch
    hangers: list[int]
    needle_hosts: list[tuple[int, float]]
    wind_bodies: list[int]
    wind_area: list[float]
    segment_length: dict[int, float]
    segment_vis_radius: dict[int, float]
    wood_radius: dict[int, float]


@dataclass
class NeedleBatch:
    half_length: float
    body: np.ndarray  # (N,) int32 host body per needle
    local: np.ndarray  # (N, 7) needle transform in the host body frame
    color: np.ndarray  # (N, 3)


def configure_builder(builder: newton.ModelBuilder) -> None:
    """Contact defaults the tree needs: a tight speculative gap and soft rigid contact.

    The default gap is far larger than a 2 mm twig; with it, most of the ~2.4k segment
    shapes register as touching each other and overflow the contact buffers.
    """
    builder.rigid_gap = 0.002
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 30.0
    builder.default_shape_cfg.mu = 0.8


class _Rods:
    """Accumulates capsule segments and rod joints, keyed by body index."""

    def __init__(self, builder: newton.ModelBuilder):
        self.builder = builder
        self.wood_cfg = builder.default_shape_cfg.copy()
        self.wood_cfg.density = WOOD_DENSITY
        self.twig_cfg = builder.default_shape_cfg.copy()
        self.twig_cfg.density = TWIG_DENSITY
        self.shape: dict[int, int] = {}
        self.length: dict[int, float] = {}
        self.vis_radius: dict[int, float] = {}
        self.wood_radius: dict[int, float] = {}
        self.modulus: dict[int, float] = {}
        self.joints: list[int] = []

    def segment(self, p0, p1, radius, color, modulus, cfg=None, kinematic=False) -> int:
        seg = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        length = float(np.linalg.norm(seg))
        d = seg / length
        q = _quat_z_to(d)
        center = 0.5 * (np.asarray(p0) + np.asarray(p1))
        body = self.builder.add_link(xform=wp.transform(wp.vec3(*center), wp.quat(*q)), is_kinematic=kinematic)
        self.shape[body] = self.builder.add_shape_capsule(
            body, radius=radius, half_height=0.5 * length, cfg=cfg or self.wood_cfg, color=color
        )
        self.length[body] = length
        self.vis_radius[body] = radius
        self.wood_radius[body] = radius
        self.modulus[body] = modulus
        return body

    def connect(self, parent: int, child: int, bend: float | None = None) -> int:
        lp, lc = self.length[parent], self.length[child]
        if bend is None:
            r = min(self.wood_radius[parent], self.wood_radius[child])
            bend = self.modulus[child] * math.pi * r**4 / 4.0 / (0.5 * (lp + lc))
        joint = self.builder.add_joint_rod(
            parent=parent,
            child=child,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * lp), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * lc), wp.quat_identity()),
            stretch_stiffness=TRUNK_STRETCH_STIFFNESS
            if self.modulus[child] == TRUNK_YOUNG_MODULUS
            else BRANCH_STRETCH_STIFFNESS,
            bend_stiffness=bend,
            bend_damping=BEND_DAMPING_RATIO * bend,
            twist_stiffness=TWIST_RATIO * bend,
            twist_damping=BEND_DAMPING_RATIO * TWIST_RATIO * bend,
            collision_filter_parent=True,
        )
        self.joints.append(joint)
        return joint

    def chain(self, parent, points, r0, r1, color, modulus, cfg=None, siblings=()) -> tuple[list[int], list[int]]:
        bodies, joints = [], []
        n = len(points) - 1
        prev = parent
        for k in range(n):
            r = r0 + (r1 - r0) * (k / max(n - 1, 1))
            body = self.segment(points[k], points[k + 1], r, color, modulus, cfg=cfg)
            joints.append(self.connect(prev, body))
            if k == 0:
                for sibling in siblings:
                    self.filter(sibling, body)
            bodies.append(body)
            prev = body
        return bodies, joints

    def filter(self, a: int, b: int) -> None:
        sa, sb = self.shape[a], self.shape[b]
        self.builder.shape_collision_filter_pairs.append((min(sa, sb), max(sa, sb)))


def _quat_z_to(d: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion (x, y, z, w) rotating +Z onto unit ``d``."""
    if d[2] < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([-d[1], d[0], 0.0, 1.0 + d[2]])
    return q / np.linalg.norm(q)


def _bent_path(start, direction, length, n, droop) -> list[np.ndarray]:
    """Points along a gently curving branch that starts along ``direction`` and arcs downward."""
    up = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    pts = [np.array(start, dtype=float)]
    for _ in range(n):
        d = d - droop * up / n
        d = d / np.linalg.norm(d)
        pts.append(pts[-1] + (length / n) * d)
    return pts


def _angle_off(azimuth: float, facing: float) -> float:
    return abs((azimuth - facing + math.pi) % (2.0 * math.pi) - math.pi)


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class _Keepout:
    """The robot's standing sector and the target's approach corridor."""

    def __init__(self, origin, facing):
        self.origin = np.asarray(origin, dtype=float)
        self.facing = facing
        self.corridor = None

    def blocks(self, points) -> bool:
        half, rmin, zmax = ROBOT_SECTOR
        for p in points:
            d = p - self.origin
            radius = math.hypot(d[0], d[1])
            if radius >= rmin and d[2] <= zmax and _angle_off(math.atan2(d[1], d[0]), self.facing) <= half:
                return True
            if self.corridor is not None and _point_segment_distance(p, *self.corridor) < CORRIDOR[1]:
                return True
        return False


def _branch_with_twigs(
    rods: _Rods,
    parent: int,
    sibling: int | None,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    r_base: float,
    modulus: float,
    keepout: _Keepout | None,
    needle_hosts: list,
) -> tuple[list[int], list[int], list[np.ndarray], list[list[int]]]:
    """One branch chain plus four twigs per node. Returns bodies, joints, nodes, twigs per fork."""
    n_seg = max(3, round(length / 0.09))
    pts = _bent_path(start, direction, length, n_seg, droop=0.15)
    branch, joints = rods.chain(parent, pts, r_base, 0.004, BRANCH_BARK, modulus, siblings=[sibling] if sibling else ())
    needle_hosts.extend((b, 0.8) for b in branch)
    side = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    side /= np.linalg.norm(side)
    up = np.array([0.0, 0.0, 1.0])
    forks: list[list[int]] = []
    for node in range(n_seg - 1):
        twig_len = 0.5 * length * (1.0 - node / n_seg)
        fork: list[int] = []
        if twig_len >= 0.06:
            roots = []
            for sgn, tilt, spread, scale in (
                (-1, 0.1, 0.7, 1.0),
                (1, 0.1, 0.7, 1.0),
                (-1, -0.35, 0.45, 0.75),
                (1, -0.35, 0.45, 0.75),
            ):
                tdir = 0.75 * direction + sgn * spread * side + tilt * up
                tpts = _bent_path(pts[node + 1], tdir, scale * twig_len, 2, droop=0.1)
                if keepout is not None and keepout.blocks(tpts[1:]):
                    continue  # pruned where the robot stands or reaches
                twig, _ = rods.chain(
                    branch[node], tpts, 0.003, 0.002, TWIG, modulus, cfg=rods.twig_cfg, siblings=[branch[node + 1]]
                )
                needle_hosts.extend((b, 1.0 + 0.25 * j) for j, b in enumerate(twig))
                roots.append(twig[0])
                fork.extend(twig)
            for a in range(len(roots)):
                for b in range(a + 1, len(roots)):
                    rods.filter(roots[a], roots[b])
        forks.append(fork)
    return branch, joints, pts, forks


def _target_branch(rods: _Rods, branch: list[int], joints: list[int], forks: list[list[int]]) -> TargetBranch:
    hang = branch[-1]
    length = rods.length[hang]
    return TargetBranch(
        segment_bodies=list(branch),
        joints=list(joints),
        tip_body=branch[-1],
        hang_body=hang,
        hang_local=np.array([0.0, 0.0, (HANG_FRACTION - 0.5) * length]),
        tangent_local=np.array([0.0, 0.0, 1.0]),
        twig_radius=rods.wood_radius[hang],
        hang_neighbours=list(forks[-1]) if forks else [],
    )


def _whorl_params(k: int) -> tuple[float, float, int, float, float]:
    """(frac, length, n_branches, r_base, elevation) for the whorl at trunk node ``k``."""
    frac = (k - FIRST_WHORL) / (TRUNK_SEGMENTS - 2 - FIRST_WHORL)
    length = 0.70 * (1.0 - frac) + 0.14
    n_branches = 6 if frac < 0.6 else 5
    r_base = 0.013 * (1.0 - frac) + 0.005
    elevation = math.radians(10.0 + 25.0 * frac)
    return frac, length, n_branches, r_base, elevation


def _direction(azimuth: float, elevation: float) -> np.ndarray:
    return np.array(
        [math.cos(azimuth) * math.cos(elevation), math.sin(azimuth) * math.cos(elevation), math.sin(elevation)]
    )


def build_tree(
    builder: newton.ModelBuilder,
    origin: tuple[float, float, float],
    facing_yaw: float,
    branch_modulus: float = BRANCH_YOUNG_MODULUS,
    seed: int = 7,
) -> Tree:
    """Build the full fir at ``origin``; its target branch points along ``facing_yaw``."""
    rng = np.random.default_rng(seed)
    rods = _Rods(builder)
    base = np.asarray(origin, dtype=float)
    keepout = _Keepout(base, facing_yaw)
    needle_hosts: list[tuple[int, float]] = []

    z = np.linspace(0.0, HEIGHT, TRUNK_SEGMENTS + 1)
    trunk: list[int] = []
    trunk_joints: list[int] = []
    for k in range(TRUNK_SEGMENTS):
        r = 0.045 + (0.010 - 0.045) * k / (TRUNK_SEGMENTS - 1)
        body = rods.segment(
            base + [0, 0, z[k]], base + [0, 0, z[k + 1]], r, BARK, TRUNK_YOUNG_MODULUS, kinematic=(k == 0)
        )
        if trunk:
            trunk_joints.append(rods.connect(trunk[-1], body))
        trunk.append(body)
    needle_hosts.extend((b, 1.1) for b in trunk[-3:])

    whorls = list(range(FIRST_WHORL, TRUNK_SEGMENTS - 1))
    target_whorl = next(k for k in whorls if TARGET_HEIGHT_RANGE[0] <= z[k] <= TARGET_HEIGHT_RANGE[1])
    golden = math.pi * (3.0 - math.sqrt(5.0))

    # Build the target whorl first so its corridor is known when pruning everything else.
    order = [target_whorl] + [k for k in whorls if k != target_whorl]
    target: TargetBranch | None = None
    branch_tips: list[tuple[int, int]] = []  # (whorl, tip body)
    all_bodies = list(trunk)
    for k in order:
        _, length, n_branches, r_base, elevation = _whorl_params(k)
        spacing = 2.0 * math.pi / n_branches
        if k == target_whorl:
            phase = facing_yaw
        else:
            phase = facing_yaw + (whorls.index(k) * golden) % spacing
            # Keep every branch out of the robot sector: nearest branch at >= sector + margin.
            nearest = min(_angle_off(phase + i * spacing, facing_yaw) for i in range(n_branches))
            if nearest < ROBOT_SECTOR[0] + math.radians(5.0):
                phase = facing_yaw + 0.5 * spacing
        roots = []
        for i in range(n_branches):
            azimuth = phase + i * spacing
            if k != target_whorl:
                azimuth += rng.uniform(-0.08, 0.08)
            branch, joints, pts, forks = _branch_with_twigs(
                rods,
                trunk[k - 1],
                trunk[k],
                base + [0, 0, z[k]],
                _direction(azimuth, elevation),
                length,
                r_base,
                branch_modulus,
                keepout if (k != target_whorl or i != 0) else None,
                needle_hosts,
            )
            all_bodies.extend(branch)
            for fork in forks:
                all_bodies.extend(fork)
            roots.append(branch[0])
            if k == target_whorl and i == 0:
                target = _target_branch(rods, branch, joints, forks)
                hang = np.asarray(pts[-2]) + HANG_FRACTION * (np.asarray(pts[-1]) - np.asarray(pts[-2]))
                tangent = (np.asarray(pts[-1]) - np.asarray(pts[-2])) / rods.length[branch[-1]]
                keepout.corridor = (hang - 0.05 * tangent, hang + CORRIDOR[0] * tangent)
            else:
                branch_tips.append((k, branch[-1]))
        for a in range(len(roots)):
            for b in range(a + 1, len(roots)):
                rods.filter(roots[a], roots[b])
    assert target is not None

    # Pre-hung ornaments on other branches, away from the target and the robot.
    hangers = _hang_ornaments(builder, rods, branch_tips, target, keepout, rng)

    top = trunk[-1]
    star_cfg = builder.default_shape_cfg.copy()
    star_cfg.density = 300.0
    builder.add_shape_sphere(
        top,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * rods.length[top] + 0.035), wp.quat_identity()),
        radius=0.035,
        cfg=star_cfg,
        color=(1.0, 0.85, 0.1),
    )
    builder.add_articulation(list(rods.joints), label="christmas_tree")

    dynamic = [b for b in all_bodies + hangers if b != trunk[0]]
    return Tree(
        bodies=all_bodies,
        joints=list(rods.joints),
        trunk_joints=trunk_joints,
        trunk_top_body=top,
        target=target,
        hangers=hangers,
        needle_hosts=needle_hosts,
        wind_bodies=dynamic,
        wind_area=[rods.length[b] * max(6.0 * rods.vis_radius[b], 0.05) for b in dynamic],
        segment_length=dict(rods.length),
        segment_vis_radius=dict(rods.vis_radius),
        wood_radius=dict(rods.wood_radius),
    )


def _hang_ornaments(
    builder, rods: _Rods, branch_tips, target: TargetBranch, keepout: _Keepout, rng, count: int = 14
) -> list[int]:
    target_hang = np.array(wp.transform_point(builder.body_q[target.hang_body], wp.vec3(*target.hang_local)))
    orn_cfg = builder.default_shape_cfg.copy()
    orn_cfg.density = 300.0
    candidates = []
    for _, tip in branch_tips:
        end = np.array(wp.transform_point(builder.body_q[tip], wp.vec3(0.0, 0.0, 0.5 * rods.length[tip])))
        hanging = [end, end - np.array([0.0, 0.0, 0.11])]
        if np.linalg.norm(end - target_hang) < 0.25 or keepout.blocks(hanging):
            continue
        candidates.append((tip, end))
    picks = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    hangers = []
    for n, index in enumerate(picks):
        tip, end = candidates[index]
        hanger = rods.segment(end, end - np.array([0.0, 0.0, 0.05]), 0.0015, (0.8, 0.8, 0.8), BRANCH_YOUNG_MODULUS)
        rods.connect(tip, hanger, bend=HANGER_BEND)
        rods.filter(tip, hanger)
        builder.add_shape_sphere(
            hanger,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.055), wp.quat_identity()),
            radius=0.03,
            cfg=orn_cfg,
            color=ORNAMENT_COLORS[n % len(ORNAMENT_COLORS)],
        )
        hangers.append(hanger)
    return hangers


def build_branch(
    builder: newton.ModelBuilder,
    root_xform: wp.transform,
    whorl: int | None = None,
    branch_modulus: float = BRANCH_YOUNG_MODULUS,
) -> tuple[TargetBranch, int]:
    """A single target-like branch with its twigs, cantilevered from a kinematic post.

    ``root_xform`` places the branch root; its local +X is the branch azimuth. ``whorl``
    defaults to the full tree's target whorl so the probe matches the hang scene. Returns the
    target description and the post body.
    """
    rods = _Rods(builder)
    z = np.linspace(0.0, HEIGHT, TRUNK_SEGMENTS + 1)
    if whorl is None:
        whorl = next(
            k
            for k in range(FIRST_WHORL, TRUNK_SEGMENTS - 1)
            if TARGET_HEIGHT_RANGE[0] <= z[k] <= TARGET_HEIGHT_RANGE[1]
        )
    _, length, _, r_base, elevation = _whorl_params(whorl)
    root = np.array(root_xform.p)
    # The post is the trunk segment the branch hangs from, made kinematic.
    trunk_r = 0.045 + (0.010 - 0.045) * (whorl - 1) / (TRUNK_SEGMENTS - 1)
    seg_len = HEIGHT / TRUNK_SEGMENTS
    post = rods.segment(root - [0, 0, seg_len], root, trunk_r, BARK, TRUNK_YOUNG_MODULUS, kinematic=True)
    azimuth_dir = np.array(wp.quat_rotate(root_xform.q, wp.vec3(1.0, 0.0, 0.0)))
    azimuth = math.atan2(azimuth_dir[1], azimuth_dir[0])
    branch, joints, _, forks = _branch_with_twigs(
        rods, post, None, root, _direction(azimuth, elevation), length, r_base, branch_modulus, None, []
    )
    builder.add_articulation(list(rods.joints), label="probe_branch")
    return _target_branch(rods, branch, joints, forks), post


def target_whorl_root_height() -> float:
    z = np.linspace(0.0, HEIGHT, TRUNK_SEGMENTS + 1)
    return float(
        next(
            z[k]
            for k in range(FIRST_WHORL, TRUNK_SEGMENTS - 1)
            if TARGET_HEIGHT_RANGE[0] <= z[k] <= TARGET_HEIGHT_RANGE[1]
        )
    )


def make_needles(tree: Tree, rng: np.random.Generator, spacing: float = 0.004, per_ring: int = 5) -> list[NeedleBatch]:
    """Scatter cone needles around host segments in each segment's frame, swept toward the tip."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    bodies, xforms, colors, sizes = [], [], [], []
    for seg, shade in tree.needle_hosts:
        length = tree.segment_length[seg]
        r = tree.segment_vis_radius[seg]
        n_rings = max(1, int(length / spacing))
        for ring in range(n_rings):
            z = -0.5 * length + (ring + rng.uniform(0.2, 0.8)) * length / n_rings
            for k in range(per_ring):
                phi = (ring * per_ring + k) * golden + rng.uniform(-0.3, 0.3)
                theta = math.radians(rng.uniform(40.0, 70.0))
                radial = np.array([math.cos(phi), math.sin(phi), 0.0])
                d = math.sin(theta) * radial + math.cos(theta) * np.array([0.0, 0.0, 1.0])
                size = int(rng.integers(0, 2))
                center = np.array([0.0, 0.0, z]) + r * radial + NEEDLE_HALF_LENGTH[size] * d
                bodies.append(seg)
                xforms.append([*center, *_quat_z_to(d)])
                g = shade * rng.uniform(0.75, 1.15)
                colors.append([0.05 * g, 0.38 * g, 0.12 * g])
                sizes.append(size)
    bodies, xforms = np.array(bodies, dtype=np.int32), np.array(xforms, dtype=np.float32)
    colors, sizes = np.clip(np.array(colors, dtype=np.float32), 0.0, 1.0), np.array(sizes)
    return [
        NeedleBatch(half, bodies[sizes == s], xforms[sizes == s], colors[sizes == s])
        for s, half in enumerate(NEEDLE_HALF_LENGTH)
    ]


@wp.kernel
def place_needles(
    needle_body: wp.array[wp.int32],
    needle_local: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    needle_xform: wp.array[wp.transform],
):
    """Render-only needles ride rigidly on their host segment."""
    i = wp.tid()
    needle_xform[i] = wp.transform_multiply(body_q[needle_body[i]], needle_local[i])


@wp.kernel
def _advance_time(time: wp.array[wp.float32], dt: float):
    time[0] = time[0] + dt


class Wind:
    """Gusting drag on tree segments, with its own on-device clock (graph-capture safe)."""

    def __init__(self, bodies, areas, strength: float, device):
        self.device = wp.get_device(device)
        self.bodies = wp.array(bodies, dtype=wp.int32, device=self.device)
        self.areas = wp.array(areas, dtype=wp.float32, device=self.device)
        self.time = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.strength = strength

    def apply(self, state, dt: float) -> None:
        """Add this substep's wind to ``state.body_f`` and advance the gust clock by ``dt``."""
        wp.launch(
            apply_wind,
            dim=self.bodies.shape[0],
            inputs=[self.bodies, self.areas, self.time, self.strength, state.body_q, state.body_qd, state.body_f],
            device=self.device,
        )
        wp.launch(_advance_time, dim=1, inputs=[self.time, dt], device=self.device)


@wp.kernel
def apply_wind(
    body_ids: wp.array[wp.int32],
    seg_area: wp.array[wp.float32],
    time: wp.array[wp.float32],
    strength: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    """Gusting quadratic drag on tree segments; zero strength disables it."""
    i = wp.tid()
    b = body_ids[i]
    t = time[0]
    gust = 0.6 + 0.4 * wp.sin(2.1 * t) * wp.sin(0.37 * t + 1.0)
    heading = 0.35 * wp.sin(0.23 * t)
    p = wp.transform_get_translation(body_q[b])
    turb = 1.0 + 0.25 * wp.sin(3.0 * p[2] + 1.7 * t) * wp.cos(2.0 * p[0] - 1.1 * t)
    u = gust * turb * strength
    rel = wp.vec3(u * wp.cos(heading), u * wp.sin(heading), 0.0) - wp.spatial_top(body_qd[b])
    body_f[b] = body_f[b] + wp.spatial_vector(0.6 * seg_area[i] * wp.length(rel) * rel, wp.vec3(0.0))
