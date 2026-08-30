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

Four upstream models are converted, selected with ``--model``: the walking model, the
all-collisions model the stand-up and roulade tasks use, the roller model, and the walking model's
gear-backlash twin. They share a skeleton and differ in geometry, so the repairs above are the same
operations with a per-model world-contact set. The backlash model needs one more repair the others
do not -- see :func:`apply_backlash_surgery`.

Usage:

.. code-block:: bash

    uv run --extra importers python scripts/tools/convert_microduck.py
    uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions
    uv run --extra importers python scripts/tools/convert_microduck.py --model rollers
    uv run --extra importers python scripts/tools/convert_microduck.py --model walk_backlash

See ``ATTRIBUTION.md`` next to the generated asset for the provenance of the source MJCF.
"""

import dataclasses
import os
import re
import xml.etree.ElementTree as ET

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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

BACKLASH_JOINT_PATTERN = re.compile(r"^passive_(?P<servo>.+)_backlash$")
"""Upstream's naming convention for the play hinges its ``add_backlash.py`` injects."""

BACKLASH_DUMMY_SUFFIX = "_dummy"
"""Suffix appended to a play hinge's name to name the intermediate body it rides on."""

BACKLASH_DUMMY_MASS = 1e-6
"""Mass [kg] of an intermediate body.

MuJoCo refuses a moving body below ``mjMINVAL``; 1e-6 kg compiles and is dynamically invisible
(9.1e-10 rad of composed-link disagreement against upstream's same-body encoding), while 1e-4 kg
already costs three orders of magnitude.
"""

BACKLASH_DUMMY_INERTIA = 1e-9
"""Diagonal inertia [kg*m^2] of an intermediate body, on all three axes.

Not smaller: Newton's ``ModelBuilder.finalize`` validates inertia against an absolute eigenvalue
floor of 1e-10 and, when it corrects, adds 1e-6 (``newton/_src/geometry/inertia.py``), so a 1e-12
dummy is silently inflated to ~1e-6 and warns once per body per environment. 1e-9 clears the floor
untouched and is still six orders of magnitude below :data:`BACKLASH_ARMATURE`, which is what sets
the play DOF's dynamics.
"""

BACKLASH_ARMATURE = 0.001
"""Rotor inertia [kg*m^2] upstream's ``backlash`` default class authors on a play DOF.

"Kept small but non-zero for solver conditioning" (``add_backlash.py``); it also dominates the dummy
body's own inertia, which is why that body's exact mass does not matter.
"""

BACKLASH_LIMIT_DEG = 1.0
"""Half of the peak-to-peak play [deg]; upstream's ``--backlash-deg 2.0`` is the total.

Authored in degrees because that is the unit the conversion writes ``physics:lowerLimit`` and
``physics:upperLimit`` in, and therefore the unit the rest of the asset is read back in. It reaches
the built model as upstream's +/-0.017453 rad.
"""

_D6_ROTATIONAL_AXIS_PREFIXES = ("limit:rotZ:", "drive:rotZ:")
"""Property prefixes the importer's D6 collapse leaves behind on a servo joint prim."""


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

    restores_backlash: bool = False
    """Whether the MJCF declares gear-play hinges the conversion has to rebuild.

    Set on the models produced by upstream's ``add_backlash.py``, whose play hinges the importer
    silently deletes; see :func:`apply_backlash_surgery`.
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
        # ``robot_walk_backlash.xml``: the walking model with upstream's ``add_backlash.py`` run over
        # it, which changes no geometry at all -- only a second, passive hinge per servo. It
        # therefore shares the walking model's world-contact set exactly.
        MicroDuckModel(
            name="walk_backlash",
            mjcf_filename="robot_walk_backlash.xml",
            world_collider_geom_names=frozenset({"left_foot_collision", "right_foot_collision"}),
            world_collider_meshes=frozenset(),
            restores_backlash=True,
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


@dataclasses.dataclass(frozen=True)
class BacklashPair:
    """One servo joint and the passive play hinge a backlash MJCF declares next to it."""

    servo: str
    """Name of the actuated MJCF joint."""

    backlash: str
    """Name of the ``passive_<servo>_backlash`` hinge sharing its body."""

    body: str
    """Name of the MJCF body both joints are declared on."""


def backlash_pairs_from_mjcf(mjcf_path: str) -> list[BacklashPair]:
    """Return the servo/backlash joint pairs a MJCF declares, in declaration order.

    The MJCF is the source of truth for which servos carry a play hinge, so nothing here is
    hard-coded: a re-run of upstream's ``add_backlash.py`` with a different ``--exclude`` set changes
    the surgery with it, and a model without play hinges yields an empty list.

    Args:
        mjcf_path: Path to the source MJCF.

    Returns:
        The pairs found, one per body that declares both a servo joint and its play hinge.

    Raises:
        ValueError: When a ``passive_<name>_backlash`` hinge has no ``<name>`` joint on its body,
            i.e. the MJCF does not encode the serial pair the surgery reproduces.
    """
    root = ET.parse(mjcf_path).getroot()
    pairs: list[BacklashPair] = []
    for body in root.iter("body"):
        names = [joint.get("name", "") for joint in body.findall("joint")]
        for name in names:
            match = BACKLASH_JOINT_PATTERN.match(name)
            if match is None:
                continue
            servo = match.group("servo")
            if servo not in names:
                raise ValueError(
                    f"'{name}' is declared on body '{body.get('name')}' but its servo joint"
                    f" '{servo}' is not, so it is not a serial play hinge."
                )
            pairs.append(BacklashPair(servo=servo, backlash=name, body=body.get("name", "")))
    return pairs


def apply_backlash_surgery(stage: Usd.Stage, mjcf_path: str) -> list[BacklashPair]:
    """Re-create a backlash MJCF's play hinges on a converted stage, and return the pairs restored.

    ``robot_walk_backlash.xml`` declares each ``passive_<servo>_backlash`` hinge as a *second* joint
    on the same MJCF body as its servo joint. MuJoCo composes those into serial DOFs sharing one
    body's inertia; UsdPhysics has no encoding for that -- two joint prims between the same body pair
    are a *parallel* loop -- so the importer groups joints by ``(body0, body1)`` and collapses each
    pair into a single D6, dropping the duplicate rotational axis. The converted asset then simply
    *is* the plain walking model, with none of the play it was generated for.

    The repair inserts one dynamically invisible intermediate body per servo, so that the two hinges
    land on different body pairs::

        parent --servo hinge--> dummy --play hinge--> child

    The dummy is colocated with the child body frame, so both joint frames are unchanged and the
    composed link pose is identical to upstream's same-body encoding. Servo joints are also retyped
    back to ``PhysicsRevoluteJoint``: the D6 type is an artifact of the collapse being undone, and
    the plain walking asset -- the conversion this one must otherwise match -- authors revolute
    joints.

    The play hinges are interleaved with the servos in the joint order the built articulation
    reports (``right_hip_yaw``, ``passive_right_hip_yaw_backlash``, ``right_hip_roll``, ...). That is
    benign for the MicroDuck tasks because every joint selection in them is by exact name, but a
    consumer that indexes joints positionally would have to be told.

    Args:
        stage: Flattened stage to edit in place.
        mjcf_path: Path to the backlash MJCF the stage was converted from.

    Returns:
        The pairs restored, one per play hinge authored.

    Raises:
        RuntimeError: When the MJCF declares no play hinges, when a servo joint it pairs is missing
            from the stage, or when that joint does not name the child body to insert one before.
    """
    pairs = backlash_pairs_from_mjcf(mjcf_path)
    if not pairs:
        raise RuntimeError(f"'{mjcf_path}' declares no 'passive_<servo>_backlash' hinges to restore.")
    joint_prims = {prim.GetName(): prim for prim in stage.TraverseAll() if prim.IsA(UsdPhysics.Joint)}

    for pair in pairs:
        servo_prim = joint_prims.get(pair.servo)
        if servo_prim is None:
            raise RuntimeError(f"The converted stage has no joint named '{pair.servo}' to insert a play hinge after.")

        joint = UsdPhysics.Joint(servo_prim)
        child_targets = joint.GetBody1Rel().GetTargets()
        if len(child_targets) != 1:
            raise RuntimeError(f"Joint '{pair.servo}' does not name exactly one child body: {child_targets}.")
        child_prim = stage.GetPrimAtPath(child_targets[0])

        dummy_prim = _author_dummy_body(stage, child_prim, f"{pair.backlash}{BACKLASH_DUMMY_SUFFIX}")
        _retype_to_revolute(servo_prim)
        # The dummy frame equals the child frame, so the servo's own child-side frame carries over
        # unchanged and the play hinge is authored on the same frame from both sides.
        local_pos = servo_prim.GetAttribute("physics:localPos1").Get()
        local_rot = servo_prim.GetAttribute("physics:localRot1").Get()
        axis = servo_prim.GetAttribute("physics:axis").Get()
        joint.GetBody1Rel().SetTargets([dummy_prim.GetPath()])

        _author_backlash_joint(
            stage,
            path=child_prim.GetPath().AppendChild(pair.backlash),
            body0=dummy_prim.GetPath(),
            body1=child_prim.GetPath(),
            axis=axis,
            local_pos=local_pos,
            local_rot=local_rot,
        )
    return pairs


def flatten_to_single_file(
    layered_usd_path: str,
    dest_path: str,
    model: MicroDuckModel = MICRODUCK_WALK_MODEL,
    mjcf_path: str | None = None,
) -> None:
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
        mjcf_path: Path of the source MJCF. Required for a model whose play hinges have to be
            rebuilt, which reads the MJCF back as the source of truth for which servos carry one.

    Raises:
        ValueError: When the model needs its play hinges rebuilt and no MJCF is given to read them
            from.
    """
    if model.restores_backlash and mjcf_path is None:
        raise ValueError(f"Model '{model.name}' needs its source MJCF to rebuild the play hinges the importer drops.")

    # ``Flatten`` hands back a layer; the repairs below need a stage to compose and edit it through.
    stage = Usd.Stage.Open(Usd.Stage.Open(layered_usd_path).Flatten())
    bind_collision_material(stage, model)
    restore_collision_masks(stage, model)
    clear_root_transform(stage)
    if model.restores_backlash:
        apply_backlash_surgery(stage, mjcf_path)
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


def _author_dummy_body(stage: Usd.Stage, child_prim: Usd.Prim, name: str) -> Usd.Prim:
    """Create a dynamically invisible rigid body colocated with a child body, and return it.

    It is authored as a sibling of the child rather than a parent of it, so the existing hierarchy --
    and the transforms of everything below the child -- is left alone; the articulation topology
    comes from the joints' body relationships, not from prim nesting.
    """
    dummy = UsdGeom.Xform.Define(stage, child_prim.GetParent().GetPath().AppendChild(name)).GetPrim()
    for property_name in child_prim.GetPropertyNames():
        if not property_name.startswith("xformOp"):
            continue
        source = child_prim.GetAttribute(property_name)
        dummy.CreateAttribute(property_name, source.GetTypeName()).Set(source.Get())

    UsdPhysics.RigidBodyAPI.Apply(dummy)
    mass_api = UsdPhysics.MassAPI.Apply(dummy)
    mass_api.CreateMassAttr().Set(BACKLASH_DUMMY_MASS)
    mass_api.CreateDensityAttr().Set(0.0)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(BACKLASH_DUMMY_INERTIA,) * 3))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return dummy


def _retype_to_revolute(joint_prim: Usd.Prim) -> None:
    """Turn a servo joint the importer collapsed into a D6 back into a plain revolute joint.

    The collapse leaves the revolute encoding (``physics:axis``, ``physics:lowerLimit`` and
    ``physics:upperLimit``) untouched and adds a ``rotZ`` limit and drive on top, so undoing it is a
    retype plus the removal of those two multiple-apply schemas.
    """
    if joint_prim.GetTypeName() == "PhysicsRevoluteJoint":
        return
    joint_prim.SetTypeName("PhysicsRevoluteJoint")
    joint_prim.RemoveAPI(UsdPhysics.LimitAPI, "rotZ")
    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "rotZ")
    for property_name in list(joint_prim.GetPropertyNames()):
        if property_name.startswith(_D6_ROTATIONAL_AXIS_PREFIXES):
            joint_prim.RemoveProperty(property_name)


def _author_backlash_joint(
    stage: Usd.Stage,
    path: Sdf.Path,
    body0: Sdf.Path,
    body1: Sdf.Path,
    axis: str,
    local_pos: Gf.Vec3f,
    local_rot: Gf.Quatf,
) -> Usd.Prim:
    """Author one play hinge, and return its prim.

    The limits are authored *active* -- a real, if tiny, range rather than a free axis -- so that the
    solver's constraint-buffer heuristics size themselves with these rows present from step 0. The
    play DOFs are on their limits by design, not occasionally.

    The gainless force drive is what the importer itself authors for a hinge no MJCF actuator drives,
    as it does for the roller model's wheels. Without it the play DOF arrives at the solver with a
    fallback effort limit instead of the unbounded one an undriven joint has.
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLocalPos0Attr().Set(local_pos)
    joint.CreateLocalRot0Attr().Set(local_rot)
    joint.CreateLocalPos1Attr().Set(local_pos)
    joint.CreateLocalRot1Attr().Set(local_rot)
    joint.CreateLowerLimitAttr().Set(-BACKLASH_LIMIT_DEG)
    joint.CreateUpperLimitAttr().Set(BACKLASH_LIMIT_DEG)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)
    prim = joint.GetPrim()
    drive = UsdPhysics.DriveAPI.Apply(prim, UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(0.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateMaxForceAttr().Set(float("inf"))
    # Newton reads the rotor inertia from the PhysX variant's attribute; the play DOF carries no dry
    # friction (upstream's ``backlash`` class sets ``frictionloss="0"``).
    prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(BACKLASH_ARMATURE)
    prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.0)
    return prim


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
            flatten_to_single_file(converter.usd_path, dest_path, model, mjcf_path)

    print(f"Converted {mjcf_path} to {dest_path}")


if __name__ == "__main__":
    main()
