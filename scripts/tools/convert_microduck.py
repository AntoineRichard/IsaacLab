# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert a MicroDuck MJCF into the single USD file shipped with ``isaaclab_assets``.

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
  material binding on the world colliders, and the MJCF ``contype``/``conaffinity`` masks;
* it clears the MJCF home height the importer bakes into the articulation root transform, so that
  an ``ArticulationCfg``'s initial state sets the spawn pose instead of offsetting it.

Three upstream models are converted, selected with ``--model``: the walking model, the
all-collisions model the stand-up and roulade tasks use, and the roller model. They share a
skeleton and differ in geometry, so the repairs above are the same operations with a per-model
world-contact set.

Usage:

.. code-block:: bash

    uv run --extra importers python scripts/tools/convert_microduck.py
    uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions
    uv run --extra importers python scripts/tools/convert_microduck.py --model rollers

See ``ATTRIBUTION.md`` next to the generated asset for the provenance of the source MJCF.
"""

import dataclasses
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

MICRODUCK_REPO_URL = "https://github.com/pollen-robotics/microduck_rl.git"
"""Upstream repository the source MJCF is taken from."""

MICRODUCK_REV = "d424a0c899f6b33cbd3daeb279913134349c0b63"
"""Pinned upstream commit.

The MJCF is regenerated from CAD upstream, so the asset is only reproducible against a fixed
revision rather than the default branch.
"""

MICRODUCK_MJCF_REPO_DIR = "src/mjlab_microduck/robot/microduck"
"""Directory holding the source MJCFs inside the upstream repository."""

MICRODUCK_USD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source",
    "isaaclab_assets",
    "data",
    "Robots",
    "PollenRobotics",
    "MicroDuck",
)
"""Default output directory, the one holding the asset's ``ATTRIBUTION.md``."""

MJCF_COLLISION_GEOM_GROUP = 3
"""MJCF ``group`` of the MicroDuck ``class="collision"`` geoms.

Every MicroDuck model draws its visual shells in group 2 and its colliders in group 3, and the
importer carries the value through as an ``mjc:group`` attribute on the converted mesh. It is what
tells a collider apart from the visual mesh built on the same source mesh -- the trunk, for
instance, has both a visual and a collision ``power_support``.
"""

_MJCF_GROUP_ATTRIBUTE = "mjc:group"
"""Attribute the importer writes the MJCF geom group to."""


@dataclasses.dataclass(frozen=True)
class MicroDuckModel:
    """One upstream MicroDuck robot model and the world-contact set its MJCF authors.

    The models share a skeleton and differ in geometry, so the conversion only has to be told which
    MJCF to read, where to write, and which of its colliders reach world contact. That last set is
    the one property the importer cannot carry (see :func:`restore_collision_masks`) and it has to be
    re-derived per model from the MJCF's own ``contype``/``conaffinity``: the walking model soles its
    contact on two named geoms, while the all-collisions and roller models let the trunk, hips,
    shins and head shells reach the ground as well.
    """

    name: str
    """Value of the ``--model`` selector."""

    mjcf_filename: str
    """Name of the source MJCF in :data:`MICRODUCK_MJCF_REPO_DIR`."""

    world_collider_geom_names: frozenset[str]
    """Names of the MJCF geoms reaching world contact, for the ones the MJCF names.

    The importer keeps a named geom's name on the Xform holding its mesh, so matching a prim or any
    of its ancestors against this set identifies the collider.
    """

    world_collider_meshes: frozenset[str]
    """Meshes of the world-contact colliders the MJCF leaves unnamed.

    ``robot_allcollisions.xml`` names only the two soles and ``robot_allcollisions_rollers.xml``
    names no geom at all, so the remaining colliders are identified by the mesh the importer names
    their prim after, gated on :data:`MJCF_COLLISION_GEOM_GROUP` so a visual mesh of the same name
    is not mistaken for one.
    """

    @property
    def mjcf_repo_path(self) -> str:
        """Path of the source MJCF inside the upstream repository."""
        return f"{MICRODUCK_MJCF_REPO_DIR}/{self.mjcf_filename}"

    @property
    def usd_path(self) -> str:
        """Default output path of the converted asset."""
        return os.path.join(MICRODUCK_USD_DIR, f"microduck_{self.name}.usd")


MICRODUCK_MODELS = {
    model.name: model
    for model in (
        # ``robot_walk.xml``: only the two named soles reach the ground; the trunk ``power_support``
        # and both shin ``leg`` colliders are ``self_collision_only``.
        MicroDuckModel(
            name="walk",
            mjcf_filename="robot_walk.xml",
            world_collider_geom_names=frozenset({"left_foot_collision", "right_foot_collision"}),
            world_collider_meshes=frozenset(),
        ),
        # ``robot_allcollisions.xml``: the walk model's skeleton with six more colliders. All of them
        # reach world contact, including the three ``jaw_soft`` head shells the roulade task lands
        # the robot's head on; the trunk's ``power_support`` shell is the only one left out, and it
        # is ``self_collision_only`` on all three models.
        MicroDuckModel(
            name="allcollisions",
            mjcf_filename="robot_allcollisions.xml",
            world_collider_geom_names=frozenset({"left_foot_collision", "right_foot_collision"}),
            world_collider_meshes=frozenset(
                {
                    "np_f970",
                    "hip_l",
                    "leg",
                    "top_head_shell",
                    "jaw",
                    "bottom_head_shell",
                }
            ),
        ),
        # ``robot_allcollisions_rollers.xml``: the all-collisions set with the two soles replaced by
        # the four tires. This MJCF names no geom at all.
        MicroDuckModel(
            name="rollers",
            mjcf_filename="robot_allcollisions_rollers.xml",
            world_collider_geom_names=frozenset(),
            world_collider_meshes=frozenset(
                {
                    "np_f970",
                    "hip_l",
                    "leg",
                    "tire",
                    "top_head_shell",
                    "jaw",
                    "bottom_head_shell",
                }
            ),
        ),
    )
}
"""The upstream models this script converts, keyed by their ``--model`` selector."""

MICRODUCK_WALK_MODEL = MICRODUCK_MODELS["walk"]
"""The model converted when ``--model`` is not given."""

MICRODUCK_MJCF_REPO_PATH = MICRODUCK_WALK_MODEL.mjcf_repo_path
"""Path of the default model's MJCF inside the upstream repository."""

MICRODUCK_USD_PATH = MICRODUCK_WALK_MODEL.usd_path
"""Default output path, next to the asset's ``ATTRIBUTION.md``."""

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


def resolve_source_mjcf(
    git_path: str | None = None, rev: str | None = None, model: MicroDuckModel | None = None
) -> str:
    """Return a local path to an upstream MicroDuck MJCF, fetching it if it is not cached.

    Args:
        git_path: Repository URL or an existing local checkout. Defaults to
            :data:`MICRODUCK_REPO_URL`.
        rev: Revision to pin to. Defaults to :data:`MICRODUCK_REV`.
        model: Model to fetch the MJCF of. Defaults to :data:`MICRODUCK_WALK_MODEL`.

    Returns:
        Local path to the model's MJCF.
    """
    from isaaclab.utils.assets import retrieve_git_asset_path  # noqa: PLC0415

    return retrieve_git_asset_path(
        git_path or MICRODUCK_REPO_URL,
        (model or MICRODUCK_WALK_MODEL).mjcf_repo_path,
        rev=MICRODUCK_REV if rev is None else rev,
    )


def bind_collision_material(stage: Usd.Stage, model: MicroDuckModel = MICRODUCK_WALK_MODEL) -> int:
    """Bind the converted physics material to the world colliders, and return how many were bound.

    The importer authors the MJCF's default friction on a ``PhysicsMaterial`` prim, but every geom is
    an instance referencing ``instances.usda``, so the bindings it writes point outside the scope of
    their reference and USD drops them ("refers to a path outside the scope of the reference ...
    Ignoring"). The feet would then slide on the backend's default friction instead of MuJoCo's. The
    binding is re-authored on the collider Xforms, which are ordinary prims in the main hierarchy
    rather than inside an instance prototype, and resolves down onto the instanced meshes.

    Args:
        stage: Flattened stage to edit in place.
        model: Model whose world-contact set the binding covers.

    Returns:
        Number of world colliders bound.

    Raises:
        RuntimeError: When the converted asset carries no physics material to bind.
    """
    material_prim = next(
        (prim for prim in stage.TraverseAll() if prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)),
        None,
    )
    if material_prim is None:
        raise RuntimeError("The converted asset has no physics material to bind to the world colliders.")

    # MuJoCo's single sliding coefficient covers both UsdPhysics coefficients.
    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.CreateStaticFrictionAttr().Set(MJCF_SLIDING_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(MJCF_SLIDING_FRICTION)

    material = UsdShade.Material(material_prim)
    bound = 0
    for prim in _world_colliders(stage, model):
        # The collider mesh itself is an instance proxy, which carries no opinions; its parent Xform
        # is the instance root, an ordinary prim, and the binding resolves down onto the mesh.
        UsdShade.MaterialBindingAPI.Apply(prim.GetParent()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, _PHYSICS_MATERIAL_PURPOSE
        )
        bound += 1
    return bound


def restore_collision_masks(stage: Usd.Stage, model: MicroDuckModel = MICRODUCK_WALK_MODEL) -> int:
    """Disable the colliders the MJCF keeps out of world contact, and return how many were disabled.

    The importer represents no ``contype``/``conaffinity`` masks -- the converted asset carries
    neither collision groups nor filtered pairs -- so the ``self_collision_only`` geoms, which
    upstream collide with each other but never with the ground, arrive as ordinary world colliders.
    Disabling them leaves exactly the MJCF's world-contact set, which is per-model: the two soles on
    the walking model, and everything but the trunk's ``self_collision_only`` shell on the
    all-collisions and roller models -- head shells included, because the roulade task rolls the
    robot over its head.

    Self-collision is not re-created in exchange: the asset is converted with ``self_collision=False``,
    so those geoms have no collision role left to play.

    Args:
        stage: Flattened stage to edit in place.
        model: Model whose world-contact set is restored.

    Returns:
        Number of collider prims disabled.
    """
    keep = {prim.GetPath() for prim in _world_colliders(stage, model)}
    targets = [
        prim.GetPath()
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.GetPath() not in keep
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


def clear_root_transform(stage: Usd.Stage) -> Gf.Vec3d:
    """Reset the articulation root's transform to identity, and return the translation it carried.

    The importer bakes the MJCF's home pose -- ``qpos0``, i.e. 0.12 m of trunk height -- into the
    articulation root's own ``xformOp:translate``. Spawning applies an asset configuration's initial
    position to the prim the asset is referenced under, so that baked-in transform *composes* with
    it: a configuration asking for 0.125 m would put the robot at 0.245 m until the first reset
    writes the default root state. Clearing it makes the initial position mean what it says, and the
    home height becomes the configuration's to own.

    Args:
        stage: Flattened stage to edit in place.

    Returns:
        The translation the root carried before it was cleared.

    Raises:
        RuntimeError: When the converted asset has no articulation root.
    """
    root_prim = next((prim for prim in stage.TraverseAll() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)), None)
    if root_prim is None:
        raise RuntimeError("The converted asset has no articulation root to clear the transform of.")

    xformable = UsdGeom.Xformable(root_prim)
    translation = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    # Emptying the op order is what makes the prim identity; the now-unused op attributes are
    # removed with it so nothing is left behind to be re-ordered back in by mistake.
    xformable.ClearXformOpOrder()
    for property_name in root_prim.GetPropertyNames():
        if property_name.startswith("xformOp:"):
            root_prim.RemoveProperty(property_name)
    return translation


def flatten_to_single_file(layered_usd_path: str, dest_path: str, model: MicroDuckModel = MICRODUCK_WALK_MODEL) -> None:
    """Compose the layered asset the MJCF importer emits into one binary USD file.

    The importer writes an interface layer that payloads geometry, materials and physics from sibling
    files. Flattening bakes the composed result -- including the selected ``"Physics"`` variant --
    into a single layer, and keeps the mesh prototypes so the instanced visual geometry is not
    duplicated. The contact properties that instancing cost the importer, and the baked-in root
    transform, are repaired on the flattened stage, where the prototypes are ordinary editable prims.

    Args:
        layered_usd_path: Path of the interface layer written by the importer.
        dest_path: Path of the single USD file to write.
        model: Model the layered asset was converted from, which fixes its world-contact set.
    """
    # ``Flatten`` hands back a layer; the repairs below need a stage to compose and edit it through.
    stage = Usd.Stage.Open(Usd.Stage.Open(layered_usd_path).Flatten())
    bind_collision_material(stage, model)
    restore_collision_masks(stage, model)
    clear_root_transform(stage)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    stage.GetRootLayer().Export(dest_path)


def _world_colliders(stage: Usd.Stage, model: MicroDuckModel) -> list[Usd.Prim]:
    """Return the collider prims of a converted model that the MJCF lets touch the ground."""
    return [
        prim
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and _is_world_collider(prim, model)
    ]


def _is_world_collider(prim: Usd.Prim, model: MicroDuckModel) -> bool:
    """Return whether a prim is, or sits under, one of a model's world-contact geoms."""
    if not model.world_collider_geom_names.isdisjoint(_ancestor_names(prim)):
        return True
    group = prim.GetAttribute(_MJCF_GROUP_ATTRIBUTE)
    return (
        prim.GetName() in model.world_collider_meshes and group.IsValid() and group.Get() == MJCF_COLLISION_GEOM_GROUP
    )


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

    parser = argparse.ArgumentParser(description="Convert a MicroDuck MJCF into a single USD file.")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to the MicroDuck MJCF. Defaults to fetching the selected model's one from the"
            f" pinned upstream commit {MICRODUCK_REV[:12]}."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MICRODUCK_MODELS),
        default=MICRODUCK_WALK_MODEL.name,
        help="Upstream robot model to convert. Selects the source MJCF, the output path and the world-contact set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to store the flattened USD file. Defaults to the model's asset shipped with isaaclab_assets.",
    )
    add_launcher_args(parser)
    args_cli = parser.parse_args()
    model = MICRODUCK_MODELS[args_cli.model]

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

    mjcf_path = os.path.abspath(args_cli.input) if args_cli.input else resolve_source_mjcf(model=model)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    dest_path = os.path.abspath(args_cli.output or model.usd_path)

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
            flatten_to_single_file(converter.usd_path, dest_path, model)

    print(f"Converted {mjcf_path} to {dest_path}")


if __name__ == "__main__":
    main()
