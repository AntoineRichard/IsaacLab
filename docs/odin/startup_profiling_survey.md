# Startup profiling survey

**Scope.** What `scripts/benchmarks/benchmark_startup.py` captures today,
what the numbers mean, and what Valhalla / comparison tooling should look
at. Content grounded in a fresh local run on `antoiner/feat/odin` at
`da072850509`.

**Audience.** Anyone reading a `startup.json` from an Odin bundle (T4
dashboard, debugging a slow startup, investigating a regression).

## 1. Pipeline overview

`benchmark_startup.py` wraps each of five phases in its own
`cProfile.Profile` session and records wall-clock time plus the top
functions by own-time:

- **`app_launch`** — `AppLauncher(...)` starts Kit and returns a running
  `SimulationApp`.
- **`python_imports`** — `import gymnasium / numpy / torch / isaaclab.envs`
  plus `isaaclab_tasks.utils.launch_simulation / resolve_task_config`.
- **`task_config`** — `resolve_task_config(task, None)` loads and
  instantiates the env config dataclass tree.
- **`env_creation`** — `launch_simulation(env_cfg)` creates the
  environment, clones it across `num_envs` instances, instantiates
  sensors / actuators.
- **`first_step`** — one `env.step(action)` call to force kernel
  compilation and first-iteration setup.

Each phase emits a schema-v1 `StartupPhase`:

```json
{
  "total_time_s": 18.4,
  "top_functions": [
    {"name": "...", "own_time_s": 1.82, "cum_time_s": 2.41, "calls": 4312}
  ]
}
```

Selection is either `top_n` (default 30, or 5 with `--whitelist_config`)
or explicit fnmatch patterns from `scripts/benchmarks/startup_whitelist.yaml`.

`startup.json` lives at `<run_id>/startup.json` in an Odin bundle.
`docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md`
documents the full schema.

## 2. Phase reference

Numbers below are from the `Isaac-Ant-Direct-v0` baseline run at the
grounding commit. Absolute timings vary between hardware; relative
breakdowns are the stable signal.

### 2.1 `app_launch`

**What it is.** `AppLauncher(headless=True)` — brings up the Kit app
runtime, loads the base Omniverse extensions, acquires GPU resources.

**Typical wall-time.** 2.72 s.

**Top functions (fresh run).**

```
  46.97 ms  lib.python3.12.copy:deepcopy
  35.71 ms  isaaclab.utils.configclass:_wrap_resolvable_strings
  25.13 ms  <built-in method builtins.isinstance>
  22.70 ms  isaaclab.utils.configclass:_custom_post_init
  14.24 ms  isaaclab.utils.configclass:_field_module_dir
```

**Commentary.** The configclass machinery (`_custom_post_init`,
`_wrap_resolvable_strings`, `_field_module_dir`) runs during Kit extension
load as default config objects are instantiated: `_custom_post_init`
deep-copies every mutable field and wraps callable strings, driving the
`deepcopy` and `isinstance` costs. These functions dominate because
extension startup registers many `@configclass` definitions. A jump in
`_custom_post_init` cumtime with no new extensions implies a deeper config
tree was added to an existing extension.

**Caveats.** Kit-subsystem load time is not visible to cProfile at
Python level — if `app_launch` total jumps but no Python function owns
the jump, the cost is inside Kit (native code). See §5.

### 2.2 `python_imports`

**What it is.** The profiled `import` block between `AppLauncher.app`
and `resolve_task_config` — torch, gymnasium, numpy, the isaaclab env
modules, and `isaaclab_tasks`.

**Typical wall-time.** 0.50 s.

**Top functions (fresh run).**

```
   7.11 ms  <built-in method builtins.__build_class__>
   6.56 ms  <built-in method builtins.isinstance>
   3.89 ms  <built-in method builtins.getattr>
   3.22 ms  lib.python3.12.copy:deepcopy
   0.80 ms  ...isaaclab_tasks.utils.importer:_walk_packages
```

**Commentary.** The dominant cost is Python's standard import machinery:
`__build_class__` constructing class bodies, `_find_and_load` resolving
modules, and `_walk_packages` scanning `isaaclab_tasks` sub-packages for
auto-registration. No single IsaacLab function owns more than 1 ms here;
the phase is import-I/O bound, not compute bound. A regression in this
phase most likely means a new package with heavy top-level side effects
was added to the import chain.

**Caveats.** First-time imports after a package update may show cache
population (`.pyc` compilation) inflating times.

### 2.3 `task_config`

**What it is.** `resolve_task_config(task_id, None)` — the gym registry
lookup + env cfg instantiation, including all `@configclass`
`__post_init__` chains.

**Typical wall-time.** 0.48 s.

**Top functions (fresh run).**

```
  55.79 ms  lib.python3.12.copy:deepcopy
  23.13 ms  <built-in method builtins.isinstance>
  22.66 ms  <method 'get' of 'dict' objects>
   6.76 ms  <built-in method builtins.id>
   0.65 ms  isaaclab.utils.configclass:configclass
```

**Commentary.** Dominated by `deepcopy` and dict/isinstance calls inside
`_custom_post_init` — every nested config field is deep-copied and
string-wrapped at instantiation time. The `configclass` decorator itself
is cheap (0.65 ms own-time) but orchestrates the chain. A sudden jump in
`deepcopy` own-time here means the config tree gained more deeply nested
mutable fields.

**Caveats.** Deeply nested `PresetCfg` resolution (e.g. on tasks with a
`newton` preset) can skew this phase's total independently of the top
Python functions.

### 2.4 `env_creation`

**What it is.** `launch_simulation(env_cfg)` — simulation context
initialisation, scene creation, env cloning, sensor / actuator / event
wiring.

**Typical wall-time.** 6.23 s.

**Top functions (fresh run).**

```
 2407.91 ms  isaaclab.sim.utils.prims:wrapper
 1016.68 ms  client.impl.__init__:stat
  657.27 ms  isaaclab.cloner.cloner_utils:usd_replicate
  493.57 ms  isaaclab_physx.cloner.physx_replicate:attach_end_fn
   50.34 ms  ...isaaclab_tasks.direct.locomotion.locomotion_env:_compute_intermediate_values
```

**Commentary.** `prims:wrapper` is the decorator guard that validates the
USD prim and iterates over all instances before delegating to the
spawner; its 2.4 s own-time reflects repeated per-prim validation across
all cloned environments. `client.impl:stat` is a USD client filesystem
stat call during asset resolution. `usd_replicate` copies prim specs to
per-environment destinations in stage, and `attach_end_fn` wires PhysX
articulation bindings after cloning — together these three account for
the bulk of scene construction time.

**Caveats.** Cloner and USD-load timings scale non-linearly with
`num_envs`; comparing `env_creation` across different `num_envs` is not
meaningful.

### 2.5 `first_step`

**What it is.** One `env.step(sample_action)` call. Forces warp kernel
compilation, first observation compute, first reward / termination
evaluation.

**Typical wall-time.** 0.19 s.

**Top functions (fresh run).**

```
  15.04 ms  ...locomotion_env:_get_rewards
   0.25 ms  warp._src.context:launch
   0.08 ms  ...locomotion_env:_apply_action
   0.07 ms  isaaclab.actuators.actuator_pd:compute
   0.06 ms  tensors.impl.api:get_dof_velocities
```

**Commentary.** `_get_rewards` owns 15 ms because it runs the full
reward computation including tensor allocations on the first call.
`warp._src.context:launch` appears with 8 calls at 0.25 ms own-time —
the per-kernel JIT compile cost does not appear in cProfile because it
occurs inside the Warp C extension; only the Python dispatch overhead is
visible here.

**Caveats.** Warp kernel compile is not visible to cProfile at the
per-function level — the cost shows up as large `warp.*:launch` own-time
on first call only.

## 3. Reading the data (cross-cutting)

**cProfile semantics.** `own_time_s` (tottime) is time spent in a
function excluding its callees; `cum_time_s` (cumtime) is the total
including callees. For hot-spot hunting use `own_time_s`; for
"which dispatch path is expensive" use `cum_time_s`. The filter in
`parse_cprofile_stats` keeps functions that are (a) inside an IsaacLab
source directory, or (b) directly called by an IsaacLab function —
enough context for diagnosis, not so much that the output is dominated
by torch internals.

**`ncalls` is now populated.** Before 2026-04-22 (isaaclab 4.6.9 and
earlier), `CProfileFunction.calls` was hardcoded to `0` due to an
upstream bug in `parse_cprofile_stats`. Bundles dated after 4.6.10 carry
the real primitive call count. When comparing across commits that
straddle the fix, ignore `calls` on older bundles.

**Whitelist vs `top_n`.** Use whitelist mode when the downstream
consumer needs a stable schema across commits (dashboards with
hard-coded series names). Use `top_n` when you want to catch newly
dominant functions — e.g. after a refactor that introduces a hot path.
A whitelist pattern that matches nothing emits a placeholder row
`(pattern, 0.0, 0.0, 0)` so the dashboard key is stable even when the
function disappears.

**Comparing across commits.** Total per-phase time is the stable
headline metric. Per-function own-time is noisy within-phase; apply a
median or mean-of-N smoothing over repeated runs before alerting on a
regression. `ncalls` is deterministic for most phases and makes a good
"did the call graph change" signal.

**Comparing across backends (PhysX vs Newton).** Whole-phase totals are
the best first-order signal. Inside-phase top functions diverge too
much for direct row-level comparison (different physics code paths);
compare at the phase-total level, then drill into per-function only if
a total diverges.

**Resource caveats.** After 2026-04-22, `Resources.*.peak` in
`training.json` is the real running-max of RSS and GPU memory during
training. Bundles dated 4.6.9 or earlier carry `peak == mean` — do not
use `peak` as a "max seen" signal on those.

## 4. Whitelist recommendations

The committed `scripts/benchmarks/startup_whitelist.yaml` provides
explicit patterns for **3** of the five phases and lets **2** fall through
to `top_n` (reasons inline in the YAML).

**When to add a pattern.** If a phase's regression or cross-commit
comparison is blocked by the `top_n` cut dropping a function you care
about, add a targeted fnmatch pattern. The placeholder-row behaviour
(zero-value row for an unmatched pattern) keeps the dashboard key
stable across runs.

**When to remove a pattern.** If a pattern consistently emits the
"matched no profiled functions" warning across runs, the function has
moved or been renamed; remove the stale pattern and pick a new one
from the fresh `top_n` output.

## 5. Open questions (seeds for T4 / later work)

Things noticed during T2.2 that are not solved here:

- **Kit-subsystem cost.** `app_launch` total can change by several
  seconds without any Python function owning the change; the cost is
  inside Kit native code. Surfacing it probably needs per-extension
  timing at the Kit level, not cProfile.
- **Warp kernel compile time.** `warp.*:launch` shows up as a huge
  own-time on first step only; a subsequent step would show near-zero.
  A dedicated "first-call kernel compile" metric separate from the
  steady-state step time might be useful.
- **USD asset-load timing per asset.** `env_creation` rolls USD loading
  into a few opaque cProfile entries; a per-asset timing breakdown
  would help regressions that come from a heavier asset being added
  to a scene.
- **GPU memory delta per phase.** Today `Resources.*.peak` is
  training-wide. A per-phase peak delta (how much memory a phase
  added) would make startup-phase memory regressions observable.

These are not promises — they're candidates for whoever builds T4 or
a future extension to T2.2.
