# Resolved Preset Banner Design

## Objective

Retain the preset choices made while resolving a task configuration and emit a compact INFO-level startup banner.
The banner must make explicit selections such as `physics=newton_kamino`, renderer selections, domain presets, and
defaults visible without adding a second configuration traversal.

## Scope

This change applies to task configuration resolution through
`isaaclab_tasks.utils.resolve_task_config()` and `isaaclab_tasks.utils.hydra_task_config()`. It covers environment
and agent `PresetCfg` nodes reached through the active configuration tree.

Direct calls to `resolve_presets()` remain unchanged. Runtime transformations that occur after preset resolution,
such as automatic `renderer=rtx` selection by `launch_simulation()`, are not preset choices and are outside this
banner's scope.

## Resolution capture

The existing active-tree walk already has the preset path, selected field name, and replacement value at the moment
each `PresetCfg` is replaced. It will append those values to a caller-provided private collector during that same
walk. No second traversal, preset rediscovery, or comparison of resolved configuration objects is introduced.

Each captured entry contains:

- the full path, including its `env` or `agent` root;
- the selected field name, including `default` when fallback selection was used; and
- the concrete replacement type name, used to make defaults such as `default -> PhysxCfg` informative.

Entries retain active-tree traversal order. Chained `PresetCfg` nodes at the same path retain separate entries rather
than overwriting one another.

## Private metadata

After both environment and agent presets resolve successfully, the collector is frozen as an immutable tuple and
attached to the resolved environment configuration under `__resolved_presets__`. The attribute is intentionally
private and is not exported as supported public API.

The double-underscore prefix is required because Isaac Lab's `class_to_dict()` currently serializes single-underscore
instance attributes but excludes names beginning with `__`. Consequently, the metadata remains available for
internal debugging without entering Hydra's configuration, saved YAML files, or downstream config overrides.

## Logging

The Hydra utility module uses its standard Python module logger. Once all preset and scalar overrides have resolved
successfully, it emits this block at INFO level, with one logger call per displayed line:

```text
---------------- Resolved task presets ----------------
env.sim.physics = newton_kamino -> NewtonCfg
env.tiled_camera = rgb -> BaseCartpoleTiledCameraCfg
env.tiled_camera.renderer_cfg = isaacsim_rtx_renderer -> IsaacRtxRendererCfg
agent = default -> dict
-------------------------------------------------------
```

The top and bottom separators have the same fixed width and isolate the banner from surrounding startup output. Each
preset occupies one line so humans and automation can scan or match individual resolutions. The banner follows the
configured INFO log level and is omitted when no presets were resolved.

Deprecated preset aliases are recorded using the canonical field name selected by the resolver. A default remains
named `default`; the replacement type communicates its concrete configuration without guessing an equivalent alias.

No banner is emitted when preset validation or later Hydra override application raises an exception.

## Performance

Capture performs one tuple append for each active `PresetCfg` already visited by the resolver. Formatting and logging
are linear in the number of resolved presets. The design adds no configuration walk, registry load, backend import,
or new dependency.

## Testing

Add exactly one small regression test in the existing Hydra utility test file. The test resolves the Cartpole task
with `physics=newton_mjwarp`, captures INFO logs, and checks the private metadata plus the banner's separators and
physics line. It stops immediately after configuration resolution: it does not launch Kit, initialize a simulator,
construct an environment, allocate GPU simulation state, or run training.

The regression test is written first and run against the unchanged resolver to confirm that it fails because the
metadata and banner do not yet exist. The same single test is then rerun after the minimal implementation. No matrix
of physics backends, renderer backends, task types, failure paths, or full training launches is added.

## Documentation and changelog

This is internal diagnostic behavior and adds no public symbol, so generated API documentation does not change. Add
one `isaaclab_tasks` patch changelog fragment under `source/isaaclab_tasks/changelog.d/` describing the new resolved
preset startup logging.

## Non-goals

- Introducing a public preset-resolution metadata API.
- Reconstructing preset names from resolved objects.
- Reporting arbitrary scalar Hydra overrides.
- Reporting runtime backend transformations that are not `PresetCfg` selections.
- Changing preset selection, validation, precedence, or CLI syntax.
