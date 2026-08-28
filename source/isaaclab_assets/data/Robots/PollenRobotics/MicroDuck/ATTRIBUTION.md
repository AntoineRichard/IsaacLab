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
uv run --extra importers python scripts/tools/convert_microduck.py \
    <checkout>/src/mjlab_microduck/robot/microduck/robot_walk.xml
```

`scripts/tools/convert_microduck.py` runs the Isaac Sim MJCF importer through
`isaaclab.sim.converters.MjcfConverter`, selects the `"physx"` entry of the generated `"Physics"`
variant set, and flattens the layered result into this single binary USD. The `"physx"` variant is
the one Isaac Lab's Newton importer reads the MJCF joint armature from, and unlike the `"mujoco"`
variant it keeps the actuator force range.

The base is left free: the model is a floating-base articulation.

## What the asset carries

The conversion is verified by `source/isaaclab_assets/test/test_microduck_asset.py`, which compares
the asset against the source MJCF. Carried over: the 14 hinge joints and their names, the
per-joint position limits, the body masses and inertias, the root spawn height, and the joint
armature and effort limits.

Not carried over, and therefore owned by the task's actuator and scene configuration:

* the position actuator gains (`kp = 0.55`, `kv = 0.0`) — the importer declines to translate them
  ("Gain and bias prm arrays are not in the expected format ... physics drive stiffness and damping
  will not be created"), so the drives are unauthored;
* the MJCF joint `damping` (0.053) and `frictionloss` (0.0048) — written only as `mjc:*`
  attributes, which are not in the schema resolver set Isaac Lab passes to Newton's USD importer;
* the contact material — the converter authors a physics material with the MJCF default friction
  (`1`, torsional `0.005`, rolling `0.0001`) but the bindings from the instanced collision meshes do
  not resolve, so the shapes fall back to the backend default friction;
* the MJCF `contype`/`conaffinity` masks — the `self_collision_only` geoms (`power_support` and both
  `leg` shells) become ordinary world colliders because no collision groups or filtered pairs are
  authored.

## Regenerating

Re-running the command above overwrites `microduck_walk.usd` in place. Re-run
`source/isaaclab_assets/test/test_microduck_asset.py` afterwards with `MICRODUCK_MJCF_PATH` pointing
at the upstream `robot_walk.xml`.
