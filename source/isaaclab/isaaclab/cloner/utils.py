# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Any

import warp as wp
from newton import AxisType, ModelBuilder
from pxr import Usd

from isaaclab.utils.timer import Timer


@Timer(name="replicate_environment", msg="Replicate environment took:", enable=True, format="ms")
def replicate_environment(
    source,
    prototype_path: str,
    path_pattern: str,
    positions: np.ndarray,
    orientations: np.ndarray,
    up_axis: AxisType = "Z",
    simplify_meshes: bool = True,
    spawn_offset: tuple[float] = (0.0, 0.0, 20.0),
    **usd_kwargs,
) -> tuple[ModelBuilder, dict[str:Any]]:
    """
    Replicates a prototype USD environment in Newton.

    Args:
        source (str | pxr.UsdStage): The file path to the USD file, or an existing USD stage instance.
        prototype_path (str): The USD path where the prototype env is defined, e.g., "/World/envs/env_0".
        path_pattern (str): The USD path pattern for replicated envs, e.g., "/World/envs/env_{}".
        num_envs (int): Number of replicas to create.
        env_spacing (tuple[float]): Environment spacing vector.
        up_axis (AxisType): The desired up-vector (should match the USD stage).
        simplify_meshes (bool): If True, simplify the meshes to reduce the number of triangles. This is useful when
        meshes with complex geometry are used as collision meshes.
        spawn_offset (tuple[float]): The offset to apply to the spawned environments.
        **usd_kwargs: Keyword arguments to pass to the USD importer (see `newton.utils.parse_usd()`).

    Returns:
        (ModelBuilder, dict): The resulting ModelBuilder containing all replicated environments and a dictionary with USD stage information.
    """

    with Timer(name="newton_env_builder", msg="Env Builder took:", enable=True, format="ms"):
        builder = ModelBuilder(up_axis=up_axis)

        #stage_info = builder.add_usd(
        #    source,
        #    ignore_paths=[prototype_path],
        #    **usd_kwargs,
        #)
        builder.add_ground_plane()

        # up_axis sanity check
        stage_up_axis = "z"
        if isinstance(stage_up_axis, str) and stage_up_axis.upper() != up_axis.upper():
            print(f"WARNING: up_axis '{up_axis}' does not match USD stage up_axis '{stage_up_axis}'")

    with Timer(name="newton_prototype_builder", msg="Prototype Builder took:", enable=True, format="ms"):
        # Get child xforms from the prototype path
        child_xforms = []
        if isinstance(source, str):
            # If source is a file path, load the stage
            stage = Usd.Stage.Open(source)
        else:
            # If source is already a stage
            stage = source

        # Get the prototype prim
        prototype_prim = stage.GetPrimAtPath(prototype_path)
        if prototype_prim.IsValid():
            # Get all child prims that are Xforms
            for child_prim in prototype_prim.GetAllChildren():
                if child_prim.GetTypeName() == "Xform":
                    child_xforms.append(child_prim.GetPath().pathString)

        #If no child xforms found, use the prototype path itself
        if not child_xforms:
            child_xforms = [prototype_path]

        prototype_builder = ModelBuilder(up_axis=up_axis)
        prototype_builder.default_joint_cfg = ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        prototype_builder.default_shape_cfg.ke = 5.0e4
        prototype_builder.default_shape_cfg.kd = 5.0e2
        prototype_builder.default_shape_cfg.kf = 1.0e3
        prototype_builder.default_shape_cfg.mu = 0.75
        prototype_builder.approximate_meshes("convex_hull")
        prototype_builder.add_usd(
            "/home/antoiner/Desktop/flattened_stage.usda",
            #source,
            root_path=prototype_path,
            xform=wp.transformf(wp.vec3(0, 0, 0.8), wp.quat_identity()),
            load_non_physics_prims=False,
            **usd_kwargs,
        )

        for i in range(6, prototype_builder.joint_dof_count):
            prototype_builder.joint_target_ke[i] = 100.0
            prototype_builder.joint_target_kd[i] = 5.0

    with Timer(name="newton_multiple_add_to_builder", msg="All add to builder took:", enable=True, format="ms"):
        # clone the prototype env with updated paths
        for i, (pos, ori) in enumerate(zip(positions, orientations)):
            body_start = builder.body_count
            shape_start = builder.shape_count
            joint_start = builder.joint_count
            articulation_start = builder.articulation_count

            builder.add_builder(
                prototype_builder, xform=wp.transform(np.array(pos) + np.array(spawn_offset), wp.quat_identity())
            )

            if i > 0:
                update_paths(
                    builder,
                    prototype_path,
                    path_pattern.format(i),
                    body_start=body_start,
                    shape_start=shape_start,
                    joint_start=joint_start,
                    articulation_start=articulation_start,
                )

    return builder, prototype_builder


def update_paths(
    builder: ModelBuilder,
    old_root: str,
    new_root: str,
    body_start: int | None = None,
    shape_start: int | None = None,
    joint_start: int | None = None,
    articulation_start: int | None = None,
) -> None:
    """Updates the paths of the builder to match the new root path.

    Args:
        builder (ModelBuilder): The builder to update.
        old_root (str): The old root path.
        new_root (str): The new root path.
        body_start (int): The start index of the bodies.
        shape_start (int): The start index of the shapes.
        joint_start (int): The start index of the joints.
        articulation_start (int): The start index of the articulations.
    """
    old_len = len(old_root)
    if body_start is not None:
        for i in range(body_start, builder.body_count):
            builder.body_key[i] = f"{new_root}{builder.body_key[i][old_len:]}"
    if shape_start is not None:
        for i in range(shape_start, builder.shape_count):
            builder.shape_key[i] = f"{new_root}{builder.shape_key[i][old_len:]}"
    if joint_start is not None:
        for i in range(joint_start, builder.joint_count):
            builder.joint_key[i] = f"{new_root}{builder.joint_key[i][old_len:]}"
    if articulation_start is not None:
        for i in range(articulation_start, builder.articulation_count):
            builder.articulation_key[i] = f"{new_root}{builder.articulation_key[i][old_len:]}"
