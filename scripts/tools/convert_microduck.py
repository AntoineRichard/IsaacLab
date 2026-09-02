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
from collections.abc import Callable

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

    mjcf_patch: Callable[[str, str], None] | None = None
    """Transform applied to the upstream MJCF before conversion, if this model patches one.

    Set only on models with no upstream MJCF of their own. The callable takes the source and
    destination paths; the destination is written into a scratch directory whose ``assets`` resolves
    to the upstream one, so ``meshdir`` keeps working.
    """

    @property
    def mjcf_repo_path(self) -> str:
        """Path of the source MJCF inside the upstream repository."""
        return f"{MICRODUCK_MJCF_REPO_DIR}/{self.mjcf_filename}"

    @property
    def usd_path(self) -> str:
        """Default output path of the converted asset."""
        return os.path.join(MICRODUCK_USD_DIR, f"microduck_{self.name}.usd")


##
# The beak hinge.
#
# The real MicroDuck has **fifteen** servos; the fifteenth is named ``mouth`` and drives a grasping
# beak. Upstream's RL models all carry fourteen actuators and weld the jaw on as a fixed geom --
# deliberately, and they say so: ``scripts/bake-duck-mesh.py`` in ``pollen-robotics/microduck`` notes
# that "``mouth`` is a servo without an MJCF joint (the jaw is a fixed geom), so it never appears in
# a bake". There is therefore no upstream model to convert, and the ``beak`` variant patches one.
#
# The three numbers below are **measured from the pinned upstream meshes**, not guessed; the
# derivation, the cross-checks and the residuals are in ``artifacts/microduck/pickplace/BEAK.md``.
##

MICRODUCK_BEAK_PIVOT = (0.00292, 0.0, -0.01800)
"""Position [m] of the jaw hinge in the ``jaw_soft`` head-body frame.

Found by intersecting the face normals of every y-perpendicular cylinder in the jaw's root region
and taking the Hough peak, independently on two meshes: ``jaw`` gives (2.91, -17.92) mm with a
6.07 mm bore and ``bottom_head_shell`` gives (2.93, -18.16) mm with a 9.19 mm one -- agreeing to
0.02 mm in x and 0.24 mm in z, with the jaw's boss seated inside the shell's larger bearing seat.
The pivot height also coincides with the ``-0.018`` every head geom is placed at, so the CAD origin
of the head parts is the hinge itself.
"""

MICRODUCK_BEAK_AXIS = (0.0, 1.0, 0.0)
"""Hinge axis [-] in the head-body frame. Every head part is symmetric about ``y = 0`` at +/-45.7 mm,
so the sagittal axis is the only candidate."""

MICRODUCK_BEAK_RANGE = (-0.08726646259971647, 0.5235987755982988)
"""Jaw travel [rad]: -5 degrees closed, +30 degrees fully open.

These are upstream's own ``MOUTH_CLOSED`` and ``MOUTH_OPEN`` from ``duck-control/src/model.rs``, and
the closed end **reproduces from the geometry**: sweeping the jaw about the measured pivot puts its
minimum gap to ``soft_mouth_top`` -- the upper surface it shuts against -- at 0.06 mm at exactly
-5.0 degrees. Two independent sources, one number. The mesh's own baked pose is therefore 5 degrees
open, which is why zero is inside the range rather than at its end.

Full gape is **17.4 mm**, which is the hard bound on what this robot can pick up.
"""

MICRODUCK_BEAK_JOINT_NAME = "mouth"
"""Name of the hinge, matching the fifteenth entry of upstream's servo wire order."""

MICRODUCK_BEAK_BODY_NAME = "beak"
"""Name of the body the jaw geoms move onto."""

MICRODUCK_BEAK_DENSITY = 1240.0
"""Density [kg/m^3] the beak's mass is taken at, PLA.

The ``jaw`` mesh is watertight at 9.36 cm^3, so this makes the beak 11.6 g of the head assembly's
188.8 g. The remainder is left on the head, and the *composite* mass, centre of mass and inertia are
preserved exactly -- splitting a body must not change the robot. That invariant is asserted by
``test_microduck_beak_asset.py`` rather than trusted.
"""


def split_beak_into_hinged_body(source_mjcf: str, dest_mjcf: str) -> None:
    """Write a copy of the all-collisions MJCF whose jaw is a hinged child body.

    Moves both ``jaw`` geoms -- the visual and the collision one -- off the ``jaw_soft`` head body
    onto a new child at :data:`MICRODUCK_BEAK_PIVOT`, hinged about :data:`MICRODUCK_BEAK_AXIS` over
    :data:`MICRODUCK_BEAK_RANGE`, and re-splits the head's inertial between the two so the composite
    is unchanged.

    No actuator is added. The mouth is not part of any policy on the real robot -- fourteen actions
    with this joint skipped -- so the action space must not grow; the asset configuration drives it
    from a separate actuator group instead.

    Args:
        source_mjcf: Path of the upstream all-collisions MJCF.
        dest_mjcf: Path to write the patched MJCF to. Its directory must resolve the same
            ``meshdir`` as the source.

    Raises:
        RuntimeError: If the source does not have the expected head body and jaw geoms.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import trimesh  # noqa: PLC0415

    tree = ET.parse(source_mjcf)
    root = tree.getroot()
    head = next((b for b in root.iter("body") if b.get("name") == "jaw_soft"), None)
    if head is None:
        raise RuntimeError(f"{source_mjcf} has no 'jaw_soft' body to hinge a beak off.")
    jaw_geoms = [g for g in head.findall("geom") if g.get("mesh") == "jaw"]
    if not jaw_geoms:
        raise RuntimeError(f"{source_mjcf} has no 'jaw' geoms on 'jaw_soft'.")

    # -- mass properties -------------------------------------------------------------------------
    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(source_mjcf)), root.find("compiler").get("meshdir", "."))
    jaw_mesh = trimesh.load(os.path.join(mesh_dir, "jaw.stl"))
    geom_pos = np.array([float(v) for v in jaw_geoms[0].get("pos", "0 0 0").split()])
    geom_quat = np.array([float(v) for v in jaw_geoms[0].get("quat", "1 0 0 0").split()])
    w, x, y, z = geom_quat / np.linalg.norm(geom_quat)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    jaw_mesh.vertices = jaw_mesh.vertices @ rot.T + geom_pos
    jaw_mesh.density = MICRODUCK_BEAK_DENSITY

    inertial = head.find("inertial")
    head_mass = float(inertial.get("mass"))
    head_com = np.array([float(v) for v in inertial.get("pos").split()])
    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in inertial.get("fullinertia").split())
    head_inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    def about_origin(mass, com, inertia):
        """Shift an inertia from a body's own centre of mass to the body frame origin."""
        return inertia + mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))

    beak_mass = float(jaw_mesh.mass)
    beak_com = np.array(jaw_mesh.center_mass)
    beak_inertia = np.array(jaw_mesh.moment_inertia)

    rest_mass = head_mass - beak_mass
    rest_com = (head_mass * head_com - beak_mass * beak_com) / rest_mass
    rest_inertia = about_origin(head_mass, head_com, head_inertia) - about_origin(beak_mass, beak_com, beak_inertia)
    rest_inertia -= rest_mass * (np.dot(rest_com, rest_com) * np.eye(3) - np.outer(rest_com, rest_com))

    def write_inertial(element, mass, com, inertia):
        element.set("mass", f"{mass:.9g}")
        element.set("pos", " ".join(f"{v:.9g}" for v in com))
        element.set(
            "fullinertia",
            " ".join(
                f"{v:.9g}"
                for v in (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
            ),
        )

    write_inertial(inertial, rest_mass, rest_com, rest_inertia)

    # -- the hinged body -------------------------------------------------------------------------
    pivot = np.array(MICRODUCK_BEAK_PIVOT)
    beak = ET.SubElement(head, "body")
    beak.set("name", MICRODUCK_BEAK_BODY_NAME)
    beak.set("pos", " ".join(f"{v:.9g}" for v in pivot))
    joint = ET.SubElement(beak, "joint")
    joint.set("name", MICRODUCK_BEAK_JOINT_NAME)
    joint.set("type", "hinge")
    joint.set("axis", " ".join(f"{v:.9g}" for v in MICRODUCK_BEAK_AXIS))
    joint.set("range", " ".join(f"{v:.17g}" for v in MICRODUCK_BEAK_RANGE))
    beak_inertial = ET.SubElement(beak, "inertial")
    write_inertial(beak_inertial, beak_mass, beak_com - pivot, beak_inertia)
    for geom in jaw_geoms:
        head.remove(geom)
        pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()]) - pivot
        geom.set("pos", " ".join(f"{v:.9g}" for v in pos))
        beak.append(geom)

    ET.indent(tree, space="  ")
    tree.write(dest_mjcf, encoding="utf-8", xml_declaration=True)


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
        # No upstream MJCF: patched from ``robot_allcollisions.xml`` by
        # :func:`split_beak_into_hinged_body`, which hinges the jaw so the beak can open. Its
        # world-contact set is the all-collisions one -- the jaw still reaches the ground, it just
        # does so on a joint now.
        MicroDuckModel(
            name="beak",
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
            mjcf_patch=split_beak_into_hinged_body,
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
    import tempfile  # noqa: PLC0415

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

    from isaaclab.physics import PhysicsCfg  # noqa: PLC0415
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: PLC0415
    from isaaclab.utils.assets import check_file_path  # noqa: PLC0415

    mjcf_path = os.path.abspath(args_cli.input) if args_cli.input else resolve_source_mjcf(model=model)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    dest_path = os.path.abspath(args_cli.output or model.usd_path)

    patch_dir = None
    if model.mjcf_patch is not None:
        # The patched MJCF has to sit somewhere its ``meshdir`` still resolves, and writing into the
        # upstream checkout would mutate a shared cache. A scratch directory with ``assets``
        # symlinked back is the same thing without the side effect.
        patch_dir = tempfile.TemporaryDirectory(prefix="microduck_beak_")
        source_dir = os.path.dirname(mjcf_path)
        os.symlink(os.path.join(source_dir, "assets"), os.path.join(patch_dir.name, "assets"))
        patched = os.path.join(patch_dir.name, f"robot_{model.name}.xml")
        model.mjcf_patch(mjcf_path, patched)
        print(f"Patched {mjcf_path} -> {patched}")
        mjcf_path = patched

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

    if patch_dir is not None:
        patch_dir.cleanup()
    print(f"Converted {mjcf_path} to {dest_path}")


if __name__ == "__main__":
    main()
