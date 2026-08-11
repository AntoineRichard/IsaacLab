# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD geometry extraction for the ASCII terminal visualizer."""

from __future__ import annotations

import re
from collections import deque
from typing import Any

import numpy as np
import trimesh

from pxr import Usd, UsdGeom

from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape

from .ascii_renderer import AsciiMesh


_GeometryKey = tuple[str, str]


def extract_scene_geometry(
    scene: Any,
    stage: Usd.Stage,
    env_index: int,
    max_faces_per_body: int,
) -> dict[_GeometryKey, AsciiMesh]:
    """Extract one environment's dynamic asset geometry in body-local frames.

    Args:
        scene: Interactive scene containing articulations and rigid objects.
        stage: USD stage containing the selected environment.
        env_index: Environment index to extract.
        max_faces_per_body: Triangle budget for each body.

    Returns:
        Meshes keyed by scene asset name and body name.
    """
    geometry: dict[_GeometryKey, AsciiMesh] = {}
    for asset_name, articulation in getattr(scene, "articulations", {}).items():
        body_names = list(getattr(articulation, "body_names", []))
        body_prims = _resolve_articulation_body_prims(stage, articulation, body_names, env_index)
        body_paths = {prim.GetPath().pathString for prim in body_prims.values()}
        for body_name in body_names:
            body_prim = body_prims.get(body_name)
            if body_prim is None:
                continue
            mesh = _extract_body_mesh(body_prim, body_paths, max_faces_per_body)
            if mesh is not None:
                geometry[(asset_name, body_name)] = mesh

    for asset_name, rigid_object in getattr(scene, "rigid_objects", {}).items():
        body_prim = _resolve_rigid_object_prim(stage, rigid_object, env_index)
        if body_prim is None:
            continue
        mesh = _extract_body_mesh(body_prim, {body_prim.GetPath().pathString}, max_faces_per_body)
        if mesh is not None:
            geometry[(asset_name, asset_name)] = mesh
    return geometry


def _resolve_articulation_body_prims(
    stage: Usd.Stage,
    articulation: Any,
    body_names: list[str],
    env_index: int,
) -> dict[str, Usd.Prim]:
    """Resolve public body names to concrete USD prims."""
    root_view = getattr(articulation, "root_view", None)
    link_paths = getattr(root_view, "link_paths", None)
    if link_paths is not None:
        try:
            env_paths = list(link_paths[env_index])
        except (IndexError, TypeError):
            env_paths = []
        backend_names = list(getattr(articulation, "backend_body_names", body_names))
        resolved = {
            body_name: stage.GetPrimAtPath(str(path))
            for body_name, path in zip(backend_names, env_paths)
            if stage.GetPrimAtPath(str(path)).IsValid()
        }
        if resolved:
            return resolved

    asset_root = _resolve_asset_root(stage, articulation, env_index)
    if asset_root is None:
        return {}
    body_name_set = set(body_names)
    return {
        prim.GetName(): prim
        for prim in _traverse_prims(asset_root)
        if prim.GetName() in body_name_set and prim.IsA(UsdGeom.Xformable)
    }


def _resolve_rigid_object_prim(stage: Usd.Stage, rigid_object: Any, env_index: int) -> Usd.Prim | None:
    """Resolve one rigid object's concrete USD prim."""
    root_view = getattr(rigid_object, "root_view", None)
    prim_paths = getattr(root_view, "prim_paths", None)
    if prim_paths is not None:
        try:
            prim = stage.GetPrimAtPath(str(prim_paths[env_index]))
            if prim.IsValid():
                return prim
        except (IndexError, TypeError):
            pass
    return _resolve_asset_root(stage, rigid_object, env_index)


def _resolve_asset_root(stage: Usd.Stage, asset: Any, env_index: int) -> Usd.Prim | None:
    """Resolve an asset config's environment-regex path."""
    prim_path = str(getattr(getattr(asset, "cfg", None), "prim_path", ""))
    concrete_path = prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_index}")
    concrete_path = re.sub(r"/envs/env_[^/]+", f"/envs/env_{env_index}", concrete_path)
    prim = stage.GetPrimAtPath(concrete_path)
    if prim.IsValid():
        return prim

    leaf_name = prim_path.rstrip("/").split("/")[-1]
    fallback = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/{leaf_name}")
    return fallback if fallback.IsValid() else None


def _extract_body_mesh(
    body_prim: Usd.Prim,
    all_body_paths: set[str],
    max_faces: int,
) -> AsciiMesh | None:
    """Extract and reduce all visible geometry owned by one rigid body."""
    body_transform = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_meshes = []
    for geometry_prim in _traverse_body_geometry(body_prim, all_body_paths):
        try:
            if geometry_prim.GetTypeName() == "Mesh":
                mesh = create_trimesh_from_geom_mesh(geometry_prim)
            else:
                mesh = create_trimesh_from_geom_shape(geometry_prim)
            geometry_transform = UsdGeom.Xformable(geometry_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            relative_transform = np.asarray(geometry_transform * body_transform.GetInverse(), dtype=np.float64).T
            mesh.apply_transform(relative_transform)
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                body_meshes.append(mesh)
        except (TypeError, ValueError):
            continue
    if not body_meshes:
        return None

    combined = trimesh.util.concatenate(body_meshes)
    combined.remove_unreferenced_vertices()
    combined = _reduce_mesh(combined, max_faces)
    vertices = tuple(tuple(float(coordinate) for coordinate in vertex) for vertex in combined.vertices)
    faces = tuple(tuple(int(index) for index in face) for face in combined.faces)
    return AsciiMesh(vertices=vertices, faces=faces, edges=_build_edge_adjacency(faces))


def _traverse_body_geometry(body_prim: Usd.Prim, all_body_paths: set[str]):
    """Yield visible render geometry without crossing into another body."""
    queue = deque([body_prim])
    body_path = body_prim.GetPath().pathString
    while queue:
        prim = queue.popleft()
        prim_path = prim.GetPath().pathString
        if prim_path != body_path and prim_path in all_body_paths:
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if imageable.ComputePurpose() == UsdGeom.Tokens.guide:
                continue
        if prim.GetTypeName() in [*PRIMITIVE_MESH_TYPES, "Mesh"]:
            yield prim
        children = prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        if children:
            queue.extend(children)


def _traverse_prims(root_prim: Usd.Prim):
    """Yield a prim subtree including instance proxies."""
    queue = deque([root_prim])
    while queue:
        prim = queue.popleft()
        yield prim
        children = prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        if children:
            queue.extend(children)


def _reduce_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Reduce dense CAD geometry while retaining its body-space envelope."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        hull = mesh.convex_hull
        if len(hull.faces) <= max_faces:
            return hull
    except (RuntimeError, ValueError):
        pass
    try:
        return mesh.bounding_box_oriented.to_mesh()
    except (RuntimeError, ValueError):
        return mesh.bounding_box.to_mesh()


def _build_edge_adjacency(
    faces: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Build undirected edge-to-face adjacency for silhouette rendering."""
    adjacency: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(first, second), max(first, second))
            adjacency.setdefault(edge, []).append(face_index)
    return tuple((first, second, tuple(face_indices)) for (first, second), face_indices in adjacency.items())
