# MicroDuck

`microduck_walk.usd` is a converted copy of the MicroDuck walk model from Pollen Robotics'
`microduck_rl` project.

## Provenance

| | |
|---|---|
| Upstream repository | <https://github.com/pollen-robotics/microduck_rl> |
| Commit | `d424a0c899f6b33cbd3daeb279913134349c0b63` (branch `develop`, 2026-08-27) |
| Source file | `src/mjlab_microduck/robot/microduck/robot_walk.xml` (plus its includes and the meshes in `assets/`) |
| License | Apache License 2.0 (see the `LICENSE` file at the root of the upstream repository) |

The upstream MJCF itself is generated from Onshape CAD by
[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot).

## Conversion

`microduck_walk.usd` is **not** committed: `.gitignore` excludes USD files from this repository
("No USD files allowed in the repo"), and `isaaclab_assets/data` is documented as local, temporary
asset hosting — released assets live on the Nucleus server. Generate the file next to this document
with:

```bash
uv run --extra importers python scripts/tools/convert_microduck.py
```

No manual checkout is needed: the script fetches `robot_walk.xml` from the pinned commit above into
a local cache. Pass a path to a `robot_walk.xml` to convert a different copy.

`scripts/tools/convert_microduck.py` runs the Isaac Sim MJCF importer through
`isaaclab.sim.converters.MjcfConverter`, selects the `"physx"` entry of the generated `"Physics"`
variant set, flattens the layered result into this single binary USD, repairs the two contact
properties the importer loses to scene-graph instancing, and clears the articulation root transform.
The `"physx"` variant is the one Isaac Lab's Newton importer reads the MJCF joint armature from, and
unlike the `"mujoco"` variant it keeps the actuator force range.

The base is left free: the model is a floating-base articulation.

## What the asset carries

The conversion is verified by `source/isaaclab_assets/test/test_microduck_asset.py`, which compares
the asset against the source MJCF. Carried over: the 14 hinge joints and their names, the per-joint
position limits, the body masses and inertias, the joint armature and effort limits, the
world-contact collider set, and the foot friction.

Two of those need the conversion script's help, because the importer loses them to scene-graph
instancing and the script repairs them on the flattened stage:

* the **contact material**. The importer authors a physics material with the MJCF default friction
  (sliding `1`, torsional `0.005`, rolling `0.0001`) but the bindings it writes from the instanced
  collision meshes point outside the scope of their reference, so USD drops them. The script rebinds
  the material on the foot collider Xforms, and authors the static friction the importer leaves at
  the schema fallback of `0` (MuJoCo has one sliding coefficient; UsdPhysics has two).
* the **`contype`/`conaffinity` masks**. The importer authors no collision groups or filtered pairs,
  so the `self_collision_only` geoms (`power_support` and both `leg` shells) would arrive as ordinary
  world colliders. The script disables them, leaving the MJCF's world-contact set: the two foot
  soles. Self-collision is not re-created in exchange — the asset is converted with
  `self_collision=False`, so those geoms have no collision role left.

A third property is deliberately dropped rather than repaired: the importer bakes the MJCF's home
pose (`qpos0`, 0.12 m of trunk height) into the articulation root's own transform. Spawning applies
an asset configuration's initial position to the prim the asset is referenced under, so that
transform would compose with it and double the spawn height. The script clears it, and the home
height becomes `MICRODUCK_CFG`'s to own.

Not carried over, and therefore owned by the task's actuator configuration:

* the position actuator gains (`kp = 0.55`, `kv = 0.0`) — the importer declines to translate them
  ("Gain and bias prm arrays are not in the expected format ... physics drive stiffness and damping
  will not be created"), so the drives are unauthored;
* the MJCF joint `damping` (0.053) and `frictionloss` (0.0048) — written only as `mjc:*` attributes,
  which are not in the schema resolver set Isaac Lab passes to Newton's USD importer.

## Regenerating

Re-running the command above overwrites `microduck_walk.usd` in place. Re-run
`source/isaaclab_assets/test/test_microduck_asset.py` afterwards; it picks up the same pinned MJCF,
or one named by `MICRODUCK_MJCF_PATH`.
