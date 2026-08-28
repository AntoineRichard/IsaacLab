# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert the MicroDuck walk MJCF into the single USD file shipped with ``isaaclab_assets``.

This wraps the same :class:`~isaaclab.sim.converters.MjcfConverter` that ``convert_mjcf.py``
drives, and only adds what that script's flags cannot express:

* it fetches the source MJCF from the pinned upstream commit, so no manual checkout is needed;
* it selects the ``"physx"`` entry of the generated ``"Physics"`` variant set, which is the only
  variant Isaac Lab's Newton importer reads the MJCF joint armature from (Newton resolves
  ``physxJoint:armature``; the ``mjc:*`` attributes of the ``"mujoco"`` variant are not in the
  resolver set Isaac Lab passes to ``ModelBuilder.add_usd``, and that variant also drops the
  drive force range);
* it flattens the layered asset the importer emits (an interface layer plus seven payloads) into
  one self-contained binary USD, so the shipped asset is a single file;
* it repairs the two contact properties the importer loses to scene-graph instancing: the physics
  material binding on the foot colliders, and the MJCF ``contype``/``conaffinity`` masks.

Usage:

.. code-block:: bash

    uv run --extra importers python scripts/tools/convert_microduck.py

See ``ATTRIBUTION.md`` next to the generated asset for the provenance of the source MJCF.
"""

import os

from pxr import Usd, UsdPhysics, UsdShade

MICRODUCK_REPO_URL = "https://github.com/pollen-robotics/microduck_rl.git"
"""Upstream repository the source MJCF is taken from."""

MICRODUCK_REV = "d424a0c899f6b33cbd3daeb279913134349c0b63"
"""Pinned upstream commit.

The MJCF is regenerated from CAD upstream, so the asset is only reproducible against a fixed
revision rather than the default branch.
"""

MICRODUCK_MJCF_REPO_PATH = "src/mjlab_microduck/robot/microduck/robot_walk.xml"
"""Path of the source MJCF inside the upstream repository."""

MICRODUCK_USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source",
    "isaaclab_assets",
    "data",
    "Robots",
    "PollenRobotics",
    "MicroDuck",
    "microduck_walk.usd",
)
"""Default output path, next to the asset's ``ATTRIBUTION.md``."""

FOOT_COLLIDER_SUFFIX = "_foot_collision"
"""Suffix of the Xforms holding the MJCF ``class="collision"`` geoms.

``robot_walk.xml`` names exactly two geoms this way, ``left_foot_collision`` and
``right_foot_collision``, and they are the only two carrying the world-colliding ``contype`` and
``conaffinity`` of 1. Every other collider in the MJCF is ``self_collision_only``
(``contype``/``conaffinity`` of 2) and never touches the ground.
"""

MJCF_SLIDING_FRICTION = 1.0
"""Sliding friction the MJCF's default geom friction applies to the foot geoms.

MuJoCo has a single sliding coefficient where UsdPhysics has a static and a dynamic one, so both are
authored from this value. The importer writes only the dynamic one and leaves the static one at the
schema fallback of 0.
"""

_PHYSICS_MATERIAL_PURPOSE = "physics"
"""Material binding purpose UsdPhysics reads, i.e. the ``material:binding:physics`` relationship.

``UsdShade`` only registers ``allPurpose``, ``preview`` and ``full``, so the token is spelled out.
"""


def resolve_source_mjcf(git_path: str | None = None, rev: str | None = None) -> str:
    """Return a local path to the upstream MicroDuck walk MJCF, fetching it if it is not cached.

    Args:
        git_path: Repository URL or an existing local checkout. Defaults to
            :data:`MICRODUCK_REPO_URL`.
        rev: Revision to pin to. Defaults to :data:`MICRODUCK_REV`.

    Returns:
        Local path to ``robot_walk.xml``.
    """
    from isaaclab.utils.assets import retrieve_git_asset_path  # noqa: PLC0415

    return retrieve_git_asset_path(
        git_path or MICRODUCK_REPO_URL,
        MICRODUCK_MJCF_REPO_PATH,
        rev=MICRODUCK_REV if rev is None else rev,
    )


def bind_foot_collision_material(stage: Usd.Stage) -> int:
    """Bind the converted physics material to the foot colliders, and return how many were bound.

    The importer authors the MJCF's default friction on a ``PhysicsMaterial`` prim, but every geom is
    an instance referencing ``instances.usda``, so the bindings it writes point outside the scope of
    their reference and USD drops them ("refers to a path outside the scope of the reference ...
    Ignoring"). The feet would then slide on the backend's default friction instead of MuJoCo's. The
    binding is re-authored on the collider Xforms, which are ordinary prims in the main hierarchy
    rather than inside an instance prototype, and resolves down onto the instanced meshes.

    Args:
        stage: Flattened stage to edit in place.

    Returns:
        Number of foot colliders bound.

    Raises:
        RuntimeError: When the converted asset carries no physics material to bind.
    """
    material_prim = next(
        (prim for prim in stage.TraverseAll() if prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)),
        None,
    )
    if material_prim is None:
        raise RuntimeError("The converted asset has no physics material to bind to the foot colliders.")

    # MuJoCo's single sliding coefficient covers both UsdPhysics coefficients.
    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.CreateStaticFrictionAttr().Set(MJCF_SLIDING_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(MJCF_SLIDING_FRICTION)

    material = UsdShade.Material(material_prim)
    bound = 0
    for prim in stage.Traverse():
        if not prim.GetName().endswith(FOOT_COLLIDER_SUFFIX):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, _PHYSICS_MATERIAL_PURPOSE
        )
        bound += 1
    return bound


def restore_collision_masks(stage: Usd.Stage) -> int:
    """Disable the colliders the MJCF keeps out of world contact, and return how many were disabled.

    The importer represents no ``contype``/``conaffinity`` masks -- the converted asset carries
    neither collision groups nor filtered pairs -- so the ``self_collision_only`` geoms, which
    upstream collide with each other but never with the ground, arrive as ordinary world colliders.
    Disabling them leaves exactly the MJCF's world-contact set: the two foot soles.

    Self-collision is not re-created in exchange: the asset is converted with ``self_collision=False``,
    so those geoms have no collision role left to play.

    Args:
        stage: Flattened stage to edit in place.

    Returns:
        Number of collider prims disabled.
    """
    targets = [
        prim.GetPath()
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and not _is_foot_collider(prim)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]

    disabled = 0
    for path in targets:
        # A prim inside an instance carries no opinions of its own, and the prototype behind it is
        # both read-only and shared with the instances that must stay enabled. Un-instancing the
        # instance root turns the composed prims into ordinary editable ones; the mesh data still
        # comes from the reference, so nothing is duplicated in the exported layer.
        instance_root = _instance_root(stage.GetPrimAtPath(path))
        if instance_root is not None:
            instance_root.SetInstanceable(False)
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr().Set(False)
        disabled += 1
    return disabled


def flatten_to_single_file(layered_usd_path: str, dest_path: str) -> None:
    """Compose the layered asset the MJCF importer emits into one binary USD file.

    The importer writes an interface layer that payloads geometry, materials and physics from sibling
    files. Flattening bakes the composed result -- including the selected ``"Physics"`` variant --
    into a single layer, and keeps the mesh prototypes so the instanced visual geometry is not
    duplicated. The contact properties that instancing cost the importer are repaired on the
    flattened stage, where the prototypes are ordinary editable prims.

    Args:
        layered_usd_path: Path of the interface layer written by the importer.
        dest_path: Path of the single USD file to write.
    """
    # ``Flatten`` hands back a layer; the repairs below need a stage to compose and edit it through.
    stage = Usd.Stage.Open(Usd.Stage.Open(layered_usd_path).Flatten())
    bind_foot_collision_material(stage)
    restore_collision_masks(stage)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    stage.GetRootLayer().Export(dest_path)


def _is_foot_collider(prim: Usd.Prim) -> bool:
    """Return whether a prim is, or sits under, one of the MJCF foot collision geoms."""
    return any(name.endswith(FOOT_COLLIDER_SUFFIX) for name in _ancestor_names(prim))


def _ancestor_names(prim: Usd.Prim) -> list[str]:
    """Return the names of a prim and all of its ancestors, up to but excluding the pseudo-root."""
    names = []
    current = prim
    while current and not current.IsPseudoRoot():
        names.append(current.GetName())
        current = current.GetParent()
    return names


def _instance_root(prim: Usd.Prim) -> Usd.Prim | None:
    """Return the closest ancestor that makes a prim part of an instance, or ``None`` when it is not."""
    current = prim
    while current and not current.IsPseudoRoot():
        if current.IsInstance():
            return current
        current = current.GetParent()
    return None


def main():
    import argparse  # noqa: PLC0415

    from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation  # noqa: PLC0415
    from isaaclab.utils.version import standalone_importers_available  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Convert the MicroDuck walk MJCF into a single USD file.")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to the MicroDuck 'robot_walk.xml'. Defaults to fetching it from the pinned upstream"
            f" commit {MICRODUCK_REV[:12]}."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=MICRODUCK_USD_PATH,
        help="Path to store the flattened USD file. Defaults to the asset shipped with isaaclab_assets.",
    )
    add_launcher_args(parser)
    args_cli = parser.parse_args()

    # Prefer kit-less: the standalone importer wheel runs the same importer without starting Kit.
    args_cli.require_kit = not standalone_importers_available()
    args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

    if args_cli.require_kit and not AppLauncher.is_available():
        raise ImportError(
            "MJCF conversion requires either the full Isaac Sim runtime or the standalone"
            " 'isaacsim-asset-isolated' importer wheel, but neither is installed."
        )

    import tempfile  # noqa: PLC0415

    from isaaclab.physics import PhysicsCfg  # noqa: PLC0415
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: PLC0415
    from isaaclab.utils.assets import check_file_path  # noqa: PLC0415

    mjcf_path = os.path.abspath(args_cli.input) if args_cli.input else resolve_source_mjcf()
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    dest_path = os.path.abspath(args_cli.output)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli):
        # The layered asset is an intermediate: only the flattened file is shipped, and keeping the
        # scratch copy out of the source tree stops a stale one from shadowing a re-conversion.
        with tempfile.TemporaryDirectory(prefix="microduck_mjcf_") as scratch_dir:
            converter = MjcfConverter(
                MjcfConverterCfg(
                    asset_path=mjcf_path,
                    usd_dir=scratch_dir,
                    force_usd_conversion=True,
                    physics_variant=MjcfConverterCfg.PhysicsVariant.PHYSX,
                )
            )
            flatten_to_single_file(converter.usd_path, dest_path)

    print(f"Converted {mjcf_path} to {dest_path}")


if __name__ == "__main__":
    main()
