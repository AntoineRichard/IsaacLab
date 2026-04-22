# Odin T2.1 — Environment Lists & Newton API Gaps Design

**Project:** Odin (multi-backend IsaacLab evaluation harness)
**Task:** T2.1 — identify what to run
**Date:** 2026-04-22
**Branch:** `antoiner/feat/odin`
**Status:** Draft — pending user review

## Context

T1 delivered the per-run benchmark pipeline (Layer 1 + Layer 2): any single
`(framework, backend, task, seed)` now produces a schema-v1 bundle via
`benchmark_{rsl_rl,skrl,startup}.py` and the Hugin/Munin runners. T3 will
dispatch many of those runs in parallel across Asgard. Before T3 can start, it
needs a concrete answer to *"which runs?"*

That is T2.1's job. The plan (`eval_plan.md`) splits the answer into three
deliverables:

1. a curated list of **PhysX** environments we want to benchmark,
2. a curated list of **Newton** environments (derived from the PhysX list by
   keeping only tasks Newton can actually run), and
3. a **Newton API gap** document explaining *why* PhysX-kept tasks couldn't
   cross over, grouped by missing API surface.

T2.1 is a one-shot (re-runnable) enumeration + curation task, not a persistent
service. It operates on the current `gym.registry` content, produces config
YAMLs and a prose doc, and hands off to T3.

T2.2 (startup profiling survey) is sibling work in the same eval_plan entry
but is out of this spec's scope — sequential per user direction.

## Goals & non-goals

**In scope:**

- Produce `tools/odin/config/physx_envs.yaml` — the curated PhysX run list,
  each row carrying `(task, framework, num_envs, max_iterations)` ready for
  T3 dispatch.
- Produce `tools/odin/config/newton_envs.yaml` — the curated Newton run list,
  derived mechanically from the kept PhysX rows that expose a `newton`
  physics preset.
- Produce `docs/odin/newton_api_gaps.md` — per-gap narrative body + per-env
  appendix table identifying what Newton would need to support to unlock the
  remaining PhysX-kept tasks.
- Provide two enumeration scripts plus shared helpers under `tools/odin/`
  that generate (and re-generate, preserving manual edits) the two YAMLs
  and the gap-candidates YAML.
- One small IsaacLab-side change: promote the existing
  `_has_physics_preset` helper from `test/env_test_utils.py` to a public
  `isaaclab_tasks.utils.presets` module so Odin (and anyone else) can reuse
  it without reaching into `test/`.

**Out of scope (deferred):**

- Any vision-workflow auto-detection beyond "does the task register
  `skrl_cfg_entry_point`." If the user wants a task that has both frameworks
  to run on SKRL, they flip `framework:` manually in the YAML. A tag-based
  vision classifier can be added later.
- Plateau-based tuning of `max_iterations`. T1 already captures
  `learning.reward.series_per_iter` — calibration against that data is a
  future iteration on the YAMLs, not part of T2.1 delivery.
- **ovphysx** backend — explicitly deferred per `eval_plan.md`.
- **T2.2** (startup profiling survey) — separate spec.
- Any change to env registration, cfg structure, or `PresetCfg`. T2.1 reads
  what's there; it does not reshape it.

**Success criteria:**

1. Running `enumerate_physx_envs.py` against the current registry on
   `antoiner/feat/odin` produces a YAML with one row per `Isaac*` task,
   grouped by directory-derived type, with `framework` / `num_envs` /
   `max_iterations` auto-populated from shipped framework configs wherever
   possible. Manually filtered to the keeper set by the user.
2. Running `enumerate_newton_envs.py` against the filtered PhysX YAML
   produces `newton_envs.yaml` (kept PhysX ∩ has newton preset) and
   `newton_gap_candidates.yaml` (kept PhysX ∩ ¬ has newton preset), both
   merging cleanly with prior hand-edits.
3. `docs/odin/newton_api_gaps.md` categorizes every
   `newton_gap_candidates.yaml` row into a known gap bucket and provides a
   per-env appendix table.
4. All three committed artifacts exist and are consistent with each other:
   `newton_envs.yaml ⊆ kept(physx_envs.yaml)`, and every row in
   `newton_gap_candidates.yaml` appears in the gap doc's appendix.

## Architecture — data flow and pipeline

Three steps. Only the first two are code; the third is hand-authoring.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1 — enumerate_physx_envs.py                                    │
│   • AppLauncher(headless) → import isaaclab_tasks                   │
│   • Iterate gym.registry for "Isaac*" task_specs                    │
│   • For each: derive group from entry_point, detect registered      │
│     framework entry points, pull num_envs / max_iterations from     │
│     the shipped framework cfg                                        │
│   • Write tools/odin/config/physx_envs.yaml (merge with existing)   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ (manual filter: flip keep: false on unwanted rows;
                             │  adjust framework / knobs / notes in place)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2 — enumerate_newton_envs.py                                   │
│   • Read physx_envs.yaml; filter to keep:true rows                  │
│   • For each kept row: load_cfg_from_registry(task_id) →            │
│     has_physics_preset(raw_cfg, "newton")                           │
│   • Preset present → tools/odin/config/newton_envs.yaml             │
│   • Preset absent  → tools/odin/config/newton_gap_candidates.yaml   │
│                      with suspected_gap: "tbd"                      │
│   • Both writes merge with existing YAML                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ (manual filter on newton_envs.yaml +
                             │  manual categorization in gap_candidates)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3 — hand-authoring                                             │
│   • Fill suspected_gap for every row in newton_gap_candidates.yaml  │
│     (controlled vocabulary; see §YAML schema)                       │
│   • Write docs/odin/newton_api_gaps.md:                             │
│     — per-gap body sections (narrative + count + unlock value)      │
│     — per-env appendix table (rendered from YAML)                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key pipeline properties:**

- **Both Step 1 and Step 2 launch Isaac Sim** via `AppLauncher(headless=True)`.
  Populating `gym.registry` requires `import isaaclab_tasks`, which in turn
  requires the Omniverse app. Step 2 additionally uses
  `load_cfg_from_registry` for per-task `PresetCfg` inspection. ~20 s
  overhead per run — acceptable.
- **Re-run semantics preserve manual edits.** Both scripts merge discovered
  rows against the existing YAML keyed on `task_id`. Existing rows keep
  their `keep`, `framework`, `num_envs`, `max_iterations`, and `notes`
  fields untouched; the derived fields (`entry_point`, `has_rsl_rl`,
  `has_skrl`, `status`) are refreshed. Rows that vanish from the registry
  are marked `status: stale` and left in the file — never silently dropped.
  New rows are inserted with `keep: true`, `status: new`. Scripts print a
  summary at exit: "N new, M stale, K unchanged."
- **Gap-candidates YAML is the source of truth for the gap doc's
  appendix.** Once categorized, it stays committed so future re-runs can
  merge new rows into the same categorization work.
- **The gap doc is hand-authored prose.** The per-env appendix can be
  re-rendered from `newton_gap_candidates.yaml` via a tiny inline script;
  the per-gap body is narrative and stays under manual control.

## YAML schema and gap doc structure

### `physx_envs.yaml` / `newton_envs.yaml` — identical schema

```yaml
schema_version: "1.0"
generated_at: "2026-04-22T14:00:00Z"     # UTC at last enumerate run
generator: "enumerate_physx_envs.py"     # or enumerate_newton_envs.py
groups:
  direct/ant:
    - task_id: Isaac-Ant-Direct-v0
      entry_point: isaaclab_tasks.direct.ant:AntEnv
      env_cfg_entry_point: isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg
      has_rsl_rl: true
      has_skrl: true
      framework: rsl_rl                  # auto-picked; user-overridable
      num_envs: 4096                     # from shipped framework cfg
      max_iterations: 300                # from shipped framework cfg
      keep: true                         # opt-out default
      status: current                    # current | new | stale
      notes: ""
  direct/anymal_c:
    - task_id: Isaac-Velocity-Flat-Anymal-C-Direct-v0
      ...
```

**Field semantics:**

- `group` (implicit — the key under `groups`): derived from the
  `entry_point` module path. `isaaclab_tasks.direct.ant:AntEnv` →
  `direct/ant`. `isaaclab_tasks.manager_based.locomotion.velocity.Xxx:Env` →
  `manager_based/locomotion/velocity`. Group is mechanical; no semantic
  rollup in v1.
- `has_rsl_rl` / `has_skrl`: `True` iff the task_spec kwargs contain
  `rsl_rl_cfg_entry_point` / `skrl_cfg_entry_point`. Diagnostic: surfaces
  the rare frameworkless row.
- `framework`: `rsl_rl` if `has_rsl_rl`; else `skrl` if `has_skrl`; else
  `null`. A `null` framework forces `keep: false` automatically and the
  script writes `notes: "No rsl_rl or skrl entry point registered."`
- `num_envs` / `max_iterations`: pulled from the selected framework's cfg
  via `load_shipped_training_defaults`. RSL-RL cfgs expose both directly;
  SKRL cfgs are normalized (see §Enumeration scripts for the mapping).
  Either can be `null` if the cfg doesn't advertise the field — in which
  case the script sets `keep: false` and annotates `notes`.
- `keep`: default `true` at insertion. The user flips this to `false` to
  exclude a task from T3's dispatch list. Preserved across re-runs.
- `status`: `current` when present in the latest enumeration; `new` when
  inserted this run; `stale` when absent from the registry but still in the
  YAML. `stale` rows are never deleted by the script — the user removes
  them consciously.
- `notes`: free-form. Survives re-runs. Intended for "why is this
  `keep: false`" rationale.

**Stable sort:** groups alphabetical; rows within a group by `task_id`. Makes
diffs reviewable across re-runs.

### `newton_gap_candidates.yaml`

Same envelope, additional per-row field `suspected_gap`:

```yaml
schema_version: "1.0"
generated_at: "2026-04-22T14:30:00Z"
generator: "enumerate_newton_envs.py"
groups:
  manager_based/locomotion:
    - task_id: Isaac-Velocity-Rough-Anymal-C-v0
      entry_point: ...
      framework: rsl_rl
      has_rsl_rl: true
      has_skrl: false
      suspected_gap: sdf_collision       # controlled vocabulary
      notes: "Rough terrain uses SDF colliders on the heightfield."
      status: current
```

**Controlled vocabulary for `suspected_gap`** (extensible):

- `sdf_collision` — SDF colliders (rough terrain, nut-and-bolt, …)
- `tendons` — tendon actuation
- `rough_terrain` — heightfield / procedural terrain (distinct from
  `sdf_collision` because a task might need heightfield without SDF)
- `manipulation_coverage` — manipulation surface untested on Newton
- `deformable` — deformable / softbody simulation
- `other` — does not fit the above; `notes:` then becomes mandatory
- `tbd` — not yet categorized (default emitted by the script)

Gap doc rendering treats any row still at `tbd` as an error.

### `docs/odin/newton_api_gaps.md` structure

```
# Newton API gaps blocking Odin environments

**Input:** newton_gap_candidates.yaml at <commit sha>
**Scope:** gaps blocking PhysX-kept envs from running on Newton
           (physx_envs.yaml ∩ ¬ newton_envs.yaml)

## 1. SDF collisions

Envs blocked: 6. Unlock value: high (rough-terrain locomotion family).

[narrative: what's missing, what "support" would look like, rough effort
 estimate, link to any Newton issue]

## 2. Tendons

Envs blocked: 2. Unlock value: medium.

[narrative]

## N. Other / TBD

[any suspected_gap: "other" rows inline with their notes]

---

## Appendix: per-env table

| Task | Group | Gap | Notes |
|------|-------|-----|-------|
| Isaac-Velocity-Rough-Anymal-C-v0 | manager_based/locomotion | sdf_collision | … |
| …                                | …                         | …              | … |
```

The appendix is a one-function render from `newton_gap_candidates.yaml`
(fewer than 30 lines of Python); the body is prose. The doc is committed
with the appendix already rendered.

## Enumeration scripts

### File layout

```
tools/odin/
├── common/
│   ├── manifest.py          (existing, T1)
│   ├── env_list.py          (NEW: YAML load/merge/write + training-defaults)
│   └── presets.py           (NEW: thin re-export from isaaclab_tasks.utils.presets)
├── config/
│   ├── physx_envs.yaml
│   ├── newton_envs.yaml
│   └── newton_gap_candidates.yaml
├── scripts/
│   ├── enumerate_physx_envs.py
│   └── enumerate_newton_envs.py
└── tests/
    ├── test_env_list.py               (merge/round-trip; no Isaac Sim)
    ├── test_group_derivation.py       (entry_point → group; no Isaac Sim)
    ├── test_framework_suggestion.py   ((has_rsl_rl, has_skrl) → framework)
    └── test_enumerate_integration.py  (slow; runs live registry end-to-end)
```

### `common/env_list.py` — shared surface

Four functions:

- `load_env_list(path: Path) -> EnvList` — PyYAML load, validate
  `schema_version`; returns empty `EnvList` if file missing (Step 1 first
  run).
- `merge(existing: EnvList, discovered: list[EnvEntry]) -> EnvList` — re-run
  semantics as described above. Keyed on `task_id`.
- `write_env_list(path: Path, env_list: EnvList) -> None` — stable sort,
  PyYAML dump with `sort_keys=False` inside rows (the row key order is
  schema-defined); sort keys True at the groups level.
- `load_shipped_training_defaults(task_spec, framework: str) -> tuple[int | None, int | None]`
  — resolves `(num_envs, max_iterations)` from the framework's cfg entry
  point.
  - For `rsl_rl`: read the `RslRlOnPolicyRunnerCfg`-subclass; `num_envs`
    from `scene.num_envs` (env cfg), `max_iterations` from
    `agent_cfg.max_iterations`.
  - For `skrl`: read the task's SKRL cfg; `num_envs` from
    `scene.num_envs`, `max_iterations` derived as
    `trainer.timesteps // (rollouts or 1)` (exact mapping to be confirmed
    during implementation against a reference SKRL task — documented in
    the implementation plan, not the spec).
  - On any resolution failure: `(None, None)` and a logged warning. Caller
    records `keep: false` + `notes: "Could not resolve training defaults: <reason>"`.

YAML library: **PyYAML** (already an IsaacLab dep). We do not use
ruamel.yaml — comments are not part of the schema; all user decisions live
in structured fields (`keep`, `notes`, `framework`, etc.).

### `common/presets.py`

```python
from isaaclab_tasks.utils.presets import has_physics_preset

__all__ = ["has_physics_preset"]
```

Exists only so tooling doesn't reach across `isaaclab_tasks` into Odin
scripts; makes the upstream dependency explicit and swappable if Odin later
ships to its own repo.

### `scripts/enumerate_physx_envs.py`

```
1. Parse CLI: --output-path, --dry-run, --regenerate, --force
2. AppLauncher(headless=True); import isaaclab_tasks
3. existing = load_env_list(output_path)
4. discovered: list[EnvEntry] = []
   for task_spec in gym.registry.values():
     if "Isaac" not in task_spec.id:
         continue
     e = EnvEntry(
         task_id=task_spec.id,
         entry_point=task_spec.entry_point,
         env_cfg_entry_point=task_spec.kwargs.get("env_cfg_entry_point"),
         has_rsl_rl="rsl_rl_cfg_entry_point" in task_spec.kwargs,
         has_skrl="skrl_cfg_entry_point"  in task_spec.kwargs,
         group=derive_group(task_spec.entry_point),
     )
     e.framework = ("rsl_rl" if e.has_rsl_rl
                    else "skrl" if e.has_skrl
                    else None)
     if e.framework is not None:
         e.num_envs, e.max_iterations = (
             load_shipped_training_defaults(task_spec, e.framework))
     e.keep = (e.framework is not None and e.num_envs is not None
               and e.max_iterations is not None)
     if not e.keep:
         e.notes = _explain_auto_reject(e)   # "No framework", "Cfg missing num_envs", …
     discovered.append(e)
5. merged = merge(existing, discovered)
6. if args.dry_run:
       print_summary(merged); return
   else:
       write_env_list(output_path, merged)
       print_summary(merged)
```

**CLI:**

```
./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py \
    [--output-path PATH]     # default tools/odin/config/physx_envs.yaml
    [--dry-run]              # print summary, no write
    [--regenerate]           # discard existing YAML; prompts unless --force
    [--force]                # no interactive prompt
```

### `scripts/enumerate_newton_envs.py`

```
1. Parse CLI: --physx-input, --newton-output, --gap-output,
              --dry-run, --regenerate, --force
2. AppLauncher(headless=True); import isaaclab_tasks
3. physx = load_env_list(args.physx_input)
   if physx is empty: fail with "Run enumerate_physx_envs.py first."
   kept = [e for g in physx.groups.values() for e in g if e.keep]
4. existing_newton = load_env_list(args.newton_output)
   existing_gaps   = load_env_list(args.gap_output)
5. newton_discovered, gap_discovered = [], []
   for e in kept:
       raw_cfg = load_cfg_from_registry(e.task_id, "env_cfg_entry_point")
       if has_physics_preset(raw_cfg, "newton"):
           newton_discovered.append(copy.deepcopy(e))   # carry fw/knobs across
       else:
           gap_entry = copy.deepcopy(e)
           gap_entry.suspected_gap = "tbd"
           gap_discovered.append(gap_entry)
6. write_env_list(args.newton_output, merge(existing_newton, newton_discovered))
   write_env_list(args.gap_output, merge(existing_gaps, gap_discovered))
   print summary: N kept-PhysX in, M newton-supported, K gap candidates
```

`has_physics_preset` lookup covers `raw_cfg.sim.physics` being a
`PresetCfg`-shaped object; see the upstream helper's implementation.

### Error handling (both scripts)

Per-task failures never abort enumeration:

- Any exception during row construction (`load_cfg_from_registry` failure,
  attribute missing, etc.) → log `WARNING enum: <task_id>: <exc_class>:
  <one-line msg>`, emit a row with `keep: false`, `status: current`,
  `notes: "Enumeration error: <exc_class>: <msg>"`. The row still exists
  in the YAML so the user can see what went wrong and decide.
- Top-level script prints a summary at exit: *N succeeded, M errored, K
  frameworkless.* Exits `0` unless the structure itself could not be
  written.
- `enumerate_newton_envs.py` with no input PhysX YAML is the one fatal case
  — exits non-zero with a pointer to Step 1.

## Testing

**Tier 1 — fast unit tests (no Isaac Sim, run in CI):**

- `tools/odin/tests/test_env_list.py`
  - merge preserves `keep`, `framework`, `num_envs`, `max_iterations`,
    `notes` on existing rows.
  - merge marks rows absent from discovery as `status: stale`.
  - merge marks rows new-in-discovery as `status: new`, `keep: true`.
  - merge never deletes rows.
  - round-trip: `load → write → load` equality for representative YAML.
  - stable sort: groups alphabetical, rows by `task_id`.
- `tools/odin/tests/test_group_derivation.py`
  - `isaaclab_tasks.direct.ant:AntEnv` → `direct/ant`.
  - `isaaclab_tasks.manager_based.locomotion.velocity.Xxx:Env` →
    `manager_based/locomotion/velocity`.
  - `isaaclab_tasks.direct.factory.factory_env:FactoryEnv` →
    `direct/factory`.
  - entry points with no `:` or no dotted path: produces `"unknown"` and
    emits a warning (not a raise).
- `tools/odin/tests/test_framework_suggestion.py`
  - decision table across `(has_rsl_rl, has_skrl) → framework | None`.
- `source/isaaclab_tasks/test/test_presets.py`
  - `has_physics_preset(cfg_without_preset, "newton")` → `False`.
  - `has_physics_preset(cfg_with_newton_preset, "newton")` → `True`.
  - `has_physics_preset(cfg_with_other_preset, "newton")` → `False`.
  - `has_physics_preset(plain_dict, "newton")` → `False` (the existing
    dict-short-circuit).
  - `has_physics_preset(top_level_preset_wrapper, "newton")` → resolves
    via `.default` (the existing top-level unwrap path).

**Tier 2 — integration smoke (Isaac Sim, slow-marked, manual):**

- `tools/odin/tests/test_enumerate_integration.py`, `@pytest.mark.slow`.
  Steps:
  1. Run `enumerate_physx_envs.py` into a tmpdir YAML.
  2. Run `enumerate_newton_envs.py` against that YAML into tmpdir YAMLs.
  3. Assert all three YAMLs parse as schema-v1.
  4. Assert every row in `newton_envs.yaml` exists in the PhysX YAML
     with `keep: true`.
  5. Assert no row in `newton_envs.yaml` also appears in
     `newton_gap_candidates.yaml`.

**Tier 3 — manual dry-run delivery:**

The actual T2.1 deliverable. Run both scripts on `antoiner/feat/odin`,
curate the YAMLs, categorize the gap candidates, hand-author the gap doc,
commit all four artifacts.

## Upstream vs Odin split

### Lands in IsaacLab (upstream)

| Path | Purpose |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/utils/presets.py` | **New.** Public `has_physics_preset(raw_cfg, preset_name) -> bool`, promoted from `test/env_test_utils.py::_has_physics_preset`. |
| `source/isaaclab_tasks/test/env_test_utils.py` | Update `_has_physics_preset` to be a thin re-export alias from the public path. Existing tests and callers unchanged. |
| `source/isaaclab_tasks/test/test_presets.py` | **New.** Unit tests for the public helper. |
| `source/isaaclab_tasks/docs/CHANGELOG.rst` | New version entry under `Added` describing `has_physics_preset`. |
| `source/isaaclab_tasks/config/extension.toml` | Version bump to match the changelog. |

Scope check: the upstream change is additive, within a single extension
package, no public API breakage, no new dependencies.

### Lands in Odin (`tools/odin/`)

| Path | Purpose |
|---|---|
| `tools/odin/common/env_list.py` | YAML load/merge/write + `load_shipped_training_defaults`. |
| `tools/odin/common/presets.py` | Thin re-export of the upstream helper. |
| `tools/odin/scripts/enumerate_physx_envs.py` | Step 1 script. |
| `tools/odin/scripts/enumerate_newton_envs.py` | Step 2 script. |
| `tools/odin/tests/test_env_list.py` | Merge / round-trip unit tests. |
| `tools/odin/tests/test_group_derivation.py` | Entry-point → group tests. |
| `tools/odin/tests/test_framework_suggestion.py` | Framework-selection tests. |
| `tools/odin/tests/test_enumerate_integration.py` | Slow-marked integration test. |
| `tools/odin/README.md` | Update: document the two enumeration commands and the filter protocol. |

### Deliverable artifacts (committed)

| Path | Purpose |
|---|---|
| `tools/odin/config/physx_envs.yaml` | PhysX-kept list, user-filtered. |
| `tools/odin/config/newton_envs.yaml` | Newton-kept list, user-filtered. |
| `tools/odin/config/newton_gap_candidates.yaml` | Categorized gap source-of-truth. |
| `docs/odin/newton_api_gaps.md` | Human-authored narrative + appendix. |

### Architecture doc update (`docs/odin/architecture.md`)

- §6 task map: T2.1 `✅`, link this spec.
- §9 change log: entry dated today noting the three committed lists, the
  gap doc, and the promoted upstream helper.

Update lands in the same commit as the deliverables, per the architecture
doc's self-imposed rule.

## Verification gates

- `./isaaclab.sh -f` clean before any commit.
- `./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_presets.py`
  — sequential.
- `./isaaclab.sh -p -m pytest tools/odin/tests/ -m 'not slow'`
  — sequential.
- Manual: run both enumeration scripts end-to-end in headless, verify the
  three YAMLs populate correctly, curate them, hand-author
  `newton_api_gaps.md`.
- Pre-commit workflow: modify files → `./isaaclab.sh -f` → review
  reformatted output → stage → `./isaaclab.sh -f` again → commit.

## Open questions (to resolve during implementation)

- **SKRL `max_iterations` mapping.** SKRL cfgs don't expose
  `max_iterations` as a first-class field; the actual "how many learning
  iterations" is implicit in `trainer.timesteps`, `rollouts`, and
  `batch_size`. The exact normalization rule needs to be confirmed against
  a reference SKRL task (likely Ant-Direct-v0 or Cartpole-Direct-v0) during
  implementation. Documented in the implementation plan.
- **Entry-point → group derivation corner cases.** Tasks with deeply
  nested paths (e.g. `manager_based.locomotion.velocity.config.anymal_c`)
  may produce groups too fine to be useful. The implementation plan should
  either cap depth at 3 components or verify that the actual registry
  entries don't hit this.
- **`PresetCfg` schema shape.** The existing `_has_physics_preset` walks
  `raw_cfg.sim.physics`. If any Odin-candidate task uses a different
  attribute layout (e.g. `raw_cfg.physics` directly), the helper needs to
  be generalized or the T2.1 spec acknowledges the edge case.
- **Pre-existing T1 dry-run bundle bug.** `odin_runs/` contains four
  bundles from T1 with stale `tb/` copies and identical reward series
  across backends (see project memory). This is T1's concern, not T2.1's;
  T2.1 does not consume those bundles. Noting here so the implementation
  plan doesn't accidentally depend on them.

These are execution-time questions, not design-time questions — the
schema, pipeline, and deliverables do not change based on how they
resolve.
