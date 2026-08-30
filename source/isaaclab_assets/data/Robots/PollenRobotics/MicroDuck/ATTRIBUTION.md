# MicroDuck

`microduck_walk.usd`, `microduck_allcollisions.usd`, `microduck_rollers.usd` and
`microduck_walk_backlash.usd` are converted copies of four MicroDuck robot models from Pollen
Robotics' `microduck_rl` project.

## Provenance

| | |
|---|---|
| Upstream repository | <https://github.com/pollen-robotics/microduck_rl> |
| Commit | `d424a0c899f6b33cbd3daeb279913134349c0b63` (branch `develop`, 2026-08-27) |
| Source files | `src/mjlab_microduck/robot/microduck/robot_walk.xml`, `robot_allcollisions.xml`, `robot_allcollisions_rollers.xml` and `robot_walk_backlash.xml` (plus their includes and the meshes in `assets/`) |
| License | Apache License 2.0 (see the `LICENSE` file at the root of the upstream repository) |

The upstream MJCF itself is generated from Onshape CAD by
[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot).

## Conversion

The assets are **not** committed: `.gitignore` excludes USD files from this repository
("No USD files allowed in the repo"), and `isaaclab_assets/data` is documented as local, temporary
asset hosting — released assets live on the Nucleus server. Generate them next to this document
with:

```bash
uv run --extra importers python scripts/tools/convert_microduck.py
uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions
uv run --extra importers python scripts/tools/convert_microduck.py --model rollers
uv run --extra importers python scripts/tools/convert_microduck.py --model walk_backlash
```

No manual checkout is needed: the script fetches the selected model's MJCF from the pinned commit
above into a local cache. Pass a path to an MJCF to convert a different copy.

`scripts/tools/convert_microduck.py` runs the Isaac Sim MJCF importer through
`isaaclab.sim.converters.MjcfConverter`, selects the `"physx"` entry of the generated `"Physics"`
variant set, flattens the layered result into this single binary USD, repairs the two contact
properties the importer loses to scene-graph instancing, and clears the articulation root transform.
The `"physx"` variant is the one Isaac Lab's Newton importer reads the MJCF joint armature from, and
unlike the `"mujoco"` variant it keeps the actuator force range.

The base is left free: the model is a floating-base articulation.

## What the asset carries

The conversion is verified by `source/isaaclab_assets/test/test_microduck_asset.py` (walk) and
`test_microduck_variant_assets.py` (the other three), which compare each asset against its source
MJCF. Carried over: the hinge joints and their names — 14 on the walk and all-collisions models, 18
on the roller model, whose four extra `passive_*_wheel` hinges are undriven, and 28 on the backlash
model, whose 14 extra `passive_*_backlash` hinges are undriven too — the per-joint position limits,
the body masses and inertias, the joint armature and effort limits, the world-contact collider set,
and the collider friction.

Three of those need the conversion script's help, because the importer loses them and the script
repairs them on the flattened stage. Two are lost to scene-graph instancing:

* the **contact material**. The importer authors a physics material with the MJCF default friction
  (sliding `1`, torsional `0.005`, rolling `0.0001`) but the bindings it writes from the instanced
  collision meshes point outside the scope of their reference, so USD drops them. The script rebinds
  the material on the world collider Xforms, and authors the static friction the importer leaves at
  the schema fallback of `0` (MuJoCo has one sliding coefficient; UsdPhysics has two).
* the **`contype`/`conaffinity` masks**. The importer authors no collision groups or filtered pairs,
  so every geom arrives as an ordinary world collider. The script disables the ones the MJCF keeps
  out of world contact, which is a per-model set re-derived from each MJCF rather than shared:

  | model | world-contact colliders | kept out |
  |---|---|---|
  | `robot_walk.xml` | the two named foot soles | the trunk `power_support` and both `leg` shells |
  | `robot_allcollisions.xml` | the two soles plus the trunk `np_f970`, both `hip_l` cheeks, both `leg` shells and the three `jaw_soft` head shells | the trunk `power_support` |
  | `robot_allcollisions_rollers.xml` | the same, with the two soles replaced by the four `tire` colliders | the trunk `power_support` |

  The head shells matter: they are what lets a task roll the robot over its head, and upstream's
  `FULL_COLLISION` config reads like it disables them while measurably not doing so. Self-collision
  is not re-created in exchange — the assets are converted with `self_collision=False`, so the
  disabled geoms have no collision role left.

The third is specific to `robot_walk_backlash.xml`, which upstream generates from `robot_walk.xml`
with `add_backlash.py` to model 2° of total gear play per servo:

* the **gear-play hinges**. Each `passive_<servo>_backlash` hinge is a second joint on the same MJCF
  body as its servo joint. MuJoCo composes those into serial DOFs; UsdPhysics reads two joint prims
  between one body pair as a parallel loop, so the importer collapses each pair into a single D6 and
  discards the duplicate rotational axis — leaving an asset that simply *is* the plain walking model.
  The script rebuilds each pair as `parent —servo→ dummy —play→ child`, where the dummy is a
  1e-6 kg body colocated with the child's frame, and re-authors the play hinge with upstream's
  ±1° range and 0.001 kg·m² of armature. The dummy inertia is 1e-9 kg·m², which is the smallest that
  clears Newton's inertia-validation floor; below it the value is silently replaced by one a thousand
  times larger than the play DOF's own armature. The play hinges are interleaved with the servos in
  the joint order the built articulation reports, which is only benign because every MicroDuck joint
  selection is by exact name.

A further property is deliberately dropped rather than repaired: the importer bakes the MJCF's home
pose (`qpos0`, 0.12 m of trunk height) into the articulation root's own transform. Spawning applies
an asset configuration's initial position to the prim the asset is referenced under, so that
transform would compose with it and double the spawn height. The script clears it, and the home
height becomes the articulation configuration's to own.

Not carried over, and therefore owned by the task's actuator configuration:

* the position actuator gains (`kp = 0.55`, `kv = 0.0`) — the importer declines to translate them
  ("Gain and bias prm arrays are not in the expected format ... physics drive stiffness and damping
  will not be created"), so the drives are unauthored;
* the MJCF joint `damping` (0.053) and `frictionloss` (0.0048) — written only as `mjc:*` attributes,
  which are not in the schema resolver set Isaac Lab passes to Newton's USD importer. The backlash
  model's play hinges lose their own `damping` (0.01) and their `solreflimit`/`solimplimit` the same
  way, and no actuator group owns those joints to restore them from.

## Regenerating

Re-running a command above overwrites that model's USD in place. Re-run
`source/isaaclab_assets/test/test_microduck_asset.py` and `test_microduck_variant_assets.py`
afterwards; they pick up the same pinned MJCFs, or ones named by `MICRODUCK_MJCF_PATH` (walk) and
`MICRODUCK_MJCF_DIR` (variants).
