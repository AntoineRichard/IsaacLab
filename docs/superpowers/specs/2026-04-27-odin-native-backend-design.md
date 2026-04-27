# Odin Native-Backend Routing — Design

**Status:** approved
**Date:** 2026-04-27
**Task covered:** Make Odin run tasks that have no preset system but a known
native backend. Today (post-preset-handling fix) tasks like
`Isaac-Quadcopter-Direct-v0` and `Isaac-Velocity-Flat-Anymal-C-Direct-v0`
end up classified as `preset_unsupported` even when `--backend physx` is
requested — the benchmark scripts try to inject `presets=physx` into a
task whose `env_cfg.sim.physics` is a plain `PhysxCfg` (no preset
wrapper), and Hydra rejects the unknown preset name. The cleaner outcome:
when the requested backend matches the task's native cfg type, run the
task without injection. When it doesn't match (e.g. `--backend newton`
on a physx-native task), skip the row at queue time with a distinct
`native_backend_mismatch` telemetry reason.

## 1. Motivation

The 2026-04-27 preset-handling fix routed tasks without preset wrappers
into the `preset_unsupported` runtime safety net — a strict
"yaml says it's not supported, refuse to run" stance. Spec §3 of that
fix called this out as deliberate ("no silent backend swap"), but the
shape of the rule was over-broad: it conflated two genuinely different
situations into one.

1. **`presets_available=["newton"]` + `--backend physx`.** The task has
   a preset system; physx isn't in it; running anyway would mean
   injecting `presets=physx` against a task whose explicit menu of
   options excludes physx. Refusing this is correct — the operator may
   have meant to run `--backend newton` on a list of tasks that happens
   to include some physx-only ones.

2. **`presets_available=[]` + `native_backend=physx` + `--backend physx`.**
   The task has *no* preset system at all; its `sim.physics` is a plain
   `PhysxCfg`; the operator asked for physx. There's nothing to swap —
   physx is what would run with or without injection. Refusing this is
   theatre: the runtime safety net fires `preset_unsupported` for a
   request that would have succeeded on its own.

The fix is to distinguish the two cases. Stamp `native_backend` on
every yaml row at enumeration time; gate the queue on a more nuanced
rule that only skips when the request would actually entail a swap;
update the benchmark scripts to drop preset injection when there's no
preset to inject INTO and the cfg type matches the request.

## 2. Goals

- **Native-backend tasks run when requested correctly.**
  `--backend physx` on Anymal-C Flat / Quadcopter / any other
  physx-native task with no preset system → the task runs (no
  injection, no failure).
- **Silent backend swaps still rejected.** `--backend newton` on a
  physx-native task → the row is skipped at queue time with
  `reason="native_backend_mismatch"`, distinct from the
  `preset_unsupported` reason that already exists.
- **Yaml is still the source of truth.** `EnvEntry.native_backend:
  str | None` is populated by the enumerator (introspecting
  `type(raw_cfg.sim.physics)`) and consumed by the queue filter.
- **Defense in depth at the benchmark-script layer.** When yaml drift
  causes the queue filter to greenlight a row that runtime says is
  unrunnable, the benchmark scripts catch it via the existing
  `preset_unsupported:` exit-2 path. New runtime branch only skips
  injection when the cfg type matches the request.
- **Backward compatible.** Yaml without the new field still loads
  (treated as "unknown — pass through to runtime safety net"),
  `dispatch.json` schema_version bumps 1.1 → 1.2 (additive — the
  major-match validator accepts any 1.x), `SkippedEntry.reason`
  string value space is widened additively (existing readers tolerate
  any string).

## 3. Non-goals

- **`dispatch.json` schema bumps 1.1 → 1.2 (minor, additive).** The
  major-match validator added by the preset-handling fix accepts
  any 1.x file, so prior 1.1 dispatches keep loading on resume.
  Widening the value space of `reason` (a free-form string) needs
  no schema change.
- **No env_list.yaml schema_version bump.** Same additive pattern as
  the preset-handling fix: new field defaults to None on read for
  rows that haven't been re-enumerated.
- **No silent reroute of mismatched requests.** `--backend newton` on
  a physx-native task is `native_backend_mismatch`, never "fall back
  to physx because that's what's available."
- **No flag-gating, no migration window.** One PR, additive
  throughout.
- **No T2.1 yaml hand-edits.** The native_backend field is populated
  by re-running the enumerator.
- **OvPhysx-native tasks remain unrunnable under Odin** (Odin
  dispatches only physx and newton today; `native_backend="ovphysx"`
  rows always fall into `native_backend_mismatch`). That stays the
  status quo until ovphysx wiring is delivered.
- **Dashboard rendering of `native_backend_mismatch` is out of
  scope** — T4.2 will render the existing `skipped[]` array;
  whatever string `reason` carries flows through.

## 4. Architecture overview

Same shape as the preset-handling fix, with one more field and one
more rule:

```
[Enumeration]      build_entry_from_task_spec gains a cfg-type
                   introspection step → stamps native_backend
                   alongside presets_available

[YAML schema]      EnvEntry.native_backend: str | None
                   ("physx" | "newton" | "ovphysx" | None)
                   No schema_version bump (additive, read-tolerant
                   via _entry_from_dict.setdefault — same pattern
                   as the preset_handling fix)

[Asgard queue]     _expand_env_list adds a second skip rule:
                     presets_available == []
                     AND native_backend
                     AND native_backend != backend
                       → SkippedEntry(reason="native_backend_mismatch",
                                       native_backend=row.native_backend)

[Benchmark
 scripts]          When has_physics_preset(raw_cfg, backend) == False:
                     - If type(raw_cfg.sim.physics) matches backend
                       (None or PhysxCfg → "physx";
                        NewtonCfg → "newton";
                        OvPhysxCfg → "ovphysx"):
                         skip injection silently, log [INFO]
                     - Else (drift case): existing exit 2 with
                       preset_unsupported: prefix
```

End behaviour after this fix:

| Task shape | Request | After fix |
|---|---|---|
| `presets_available=[physx, newton]` | physx or newton | run (inject named preset) — unchanged |
| `presets_available=[newton]` | physx | skip (`preset_unsupported`) — unchanged |
| `presets_available=[]`, native=physx | physx | **run natively (no injection)** |
| `presets_available=[]`, native=physx | newton | **skip (`native_backend_mismatch`)** |
| `presets_available=[]`, native=None | any | pass through to runtime safety net (catch via `preset_unsupported:` exit 2) |
| `presets_available=[]`, native=ovphysx | physx or newton | skip (`native_backend_mismatch`) |

Yaml is the **primary** filter; the benchmark-script layer is the
**safety net** for yaml drift.

## 5. Schema changes

### 5.1 `EnvEntry` (`tools/odin/common/env_list.py`)

One new field after `presets_available`:

```python
@dataclass
class EnvEntry:
    ...
    presets_available: list[str] = field(default_factory=list)
    native_backend: str | None = None
    notes: str = ""
```

`_ENTRY_FIELD_ORDER` extended (insert between `presets_available`
and `notes`).

`_entry_from_dict` gains `known.setdefault("native_backend", None)`.

`merge` propagates it as a **derived** field (refreshed each pass,
like `presets_available`):
- "new only" branch: `native_backend=new.native_backend`.
- "old + new" branch: `native_backend=new.native_backend` (in the
  "Derived / refreshed from current registry" group).
- "stale" branch: unchanged (uses `**asdict(old)` which already
  carries the field).

### 5.2 Yaml row

One new key per task, written by the enumerator:

```yaml
- task_id: Isaac-Quadcopter-Direct-v0
  ...
  presets_available: []
  native_backend: physx
  notes: ''
```

Pre-fix yaml (no `native_backend` key) loads with
`native_backend=None`.

### 5.3 `SkippedEntry` (`tools/odin/asgard/jobs.py`)

One new optional field for telemetry:

```python
@dataclass
class SkippedEntry:
    task_id: str
    framework: str
    backend: str
    seed: int
    reason: str
    presets_available: list[str] = field(default_factory=list)
    native_backend: str | None = None
```

`_skipped_to_dict` / `_skipped_from_dict` in
`tools/odin/asgard/state.py` extended to round-trip the new field.
`_skipped_from_dict` defaults missing field to `None`.

`reason` value space widens to include `"native_backend_mismatch"`
alongside the existing `"preset_unsupported"`. No enum machinery to
update — `reason` is a free-form `str` today and the aggregator's
failure-kind handling already passes any string through.

### 5.4 `dispatch.json` schema_version

**Bump 1.1 → 1.2.** Adding an optional `native_backend: str | None`
field to SkippedEntry is an additive minor change per
`docs/odin/architecture.md` §5 ("minor bump = additive changes (new
optional field)"). The major-match validator added by the
preset-handling fix accepts any 1.x file, so 1.1 dispatch.json
files keep loading on resume. `state.py:SCHEMA_VERSION` advances
to `"1.2"`; new writes carry `1.2`; pre-1.2 reads default the
missing field to `None` via `_skipped_from_dict`'s defensive
`.get(...)`.

## 6. Enumeration update

`tools/odin/common/env_list.py:build_entry_from_task_spec` already
calls `raw_cfg_loader(task_spec.id)` (added by the preset-handling
fix). We extend that block with one more derivation:

```python
def _derive_native_backend(raw_cfg) -> str | None:
    """Inspect raw_cfg.sim.physics to determine the task's native backend.

    Returns:
        - "physx" if sim.physics is None (SimulationCfg default ==
          PhysxCfg per the SimulationCfg docstring) or an instance of
          PhysxCfg.
        - "newton" if sim.physics is a NewtonCfg.
        - "ovphysx" if sim.physics is an OvPhysxCfg.
        - None if sim.physics is a PresetCfg (preset system handles
          backend selection — presets_available is the source of truth)
          or an unrecognised type.
    """
    sim = getattr(raw_cfg, "sim", None)
    if sim is None:
        return None
    physics = getattr(sim, "physics", None)
    if physics is None:
        return "physx"
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_newton.physics import NewtonCfg
    try:
        from isaaclab_ovphysx.physics import OvPhysxCfg
    except ImportError:
        OvPhysxCfg = None
    from isaaclab_tasks.utils.hydra import PresetCfg

    if isinstance(physics, PhysxCfg):
        return "physx"
    if isinstance(physics, NewtonCfg):
        return "newton"
    if OvPhysxCfg is not None and isinstance(physics, OvPhysxCfg):
        return "ovphysx"
    if isinstance(physics, PresetCfg):
        return None
    return None
```

`build_entry_from_task_spec` gains a `native_backend_fn=None` kwarg
(defaulting to `_derive_native_backend`, lazy-imported when `None`,
matching the existing `has_physics_preset_fn` injection pattern).

The function is called once per task right after the existing
`presets_available` block, and the result is passed to
`EnvEntry(...)` as `native_backend=native_backend`.

**Migration**: re-run both enumerators after the implementation lands.
Both auto-merge with the existing yaml; no flags needed.

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

The yaml diffs land in the same PR as the code so the change is
self-validating. Spot-check post-regen:
- `Isaac-Velocity-Flat-Anymal-C-Direct-v0`, `Isaac-Quadcopter-Direct-v0`
  → `native_backend: physx`.
- `Isaac-Ant-Direct-v0` (preset system) → `native_backend: null`.
- `Isaac-Velocity-Flat-G1-v0`, `Isaac-Velocity-Flat-Spot-v0`,
  `Isaac-Velocity-Flat-Unitree-Go1-v0` (newton-native per prior
  enumeration) → `native_backend: newton`.

## 7. Asgard queue-time filter

`tools/odin/asgard/jobs.py:_expand_env_list` gets a second skip rule.
Final shape:

```python
def _expand_env_list(
    yaml_path: Path,
    backend: str,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None,
) -> tuple[list[JobEntry], list[SkippedEntry]]:
    env_list = load_env_list(yaml_path)
    jobs: list[JobEntry] = []
    skipped: list[SkippedEntry] = []
    for group_rows in env_list.groups.values():
        for row in group_rows:
            if not row.keep or row.status == "stale":
                continue
            if not _apply_include_filter(row.task_id, include_filter):
                continue
            if row.framework is None or row.num_envs is None or row.max_iterations is None:
                continue

            # Rule 1 (existing): preset system says backend not supported.
            if row.presets_available and backend not in row.presets_available:
                for seed in seeds:
                    skipped.append(
                        SkippedEntry(
                            task_id=row.task_id,
                            framework=row.framework,
                            backend=backend,
                            seed=seed,
                            reason="preset_unsupported",
                            presets_available=list(row.presets_available),
                            native_backend=row.native_backend,
                        )
                    )
                continue

            # Rule 2 (new): no preset system, native_backend known and mismatching.
            if (
                not row.presets_available
                and row.native_backend is not None
                and row.native_backend != backend
            ):
                for seed in seeds:
                    skipped.append(
                        SkippedEntry(
                            task_id=row.task_id,
                            framework=row.framework,
                            backend=backend,
                            seed=seed,
                            reason="native_backend_mismatch",
                            presets_available=[],
                            native_backend=row.native_backend,
                        )
                    )
                continue

            for seed in seeds:
                run_id = _make_run_id(row.framework, backend, row.task_id, dispatch_id, seed)
                jobs.append(
                    JobEntry(
                        run_id=run_id,
                        task_id=row.task_id,
                        framework=row.framework,
                        backend=backend,
                        num_envs=row.num_envs,
                        max_iterations=row.max_iterations,
                        seed=seed,
                        bundle_dir_name=run_id,
                    )
                )
    return jobs, skipped
```

**Rule ordering**: rule 1 fires first. A task with `presets_available=["newton"]`
and `native_backend="physx"` (theoretically possible if a task wraps
Newton in a PresetCfg with no physx alternative but defaults to physx
internally) when requested as physx → `preset_unsupported`, not
`native_backend_mismatch`. Tested explicitly in the test plan
(§9.2 `test_preset_unsupported_takes_precedence_over_native`).

**`native_backend` populated on both rules' SkippedEntry**: rule 1
includes it for completeness even though the discriminator there is
`presets_available`. Rule 2 always includes it (it's the whole point
of the new reason).

**Pre-dispatch stdout block** in `tools/odin/asgard/runner.py`'s
`run_dispatch` extended: existing block groups by `(task_id, backend)`;
new block also groups by `reason`. For
`native_backend_mismatch`, the line shows `native: <X>` instead of
`available: [...]`:

```
[INFO] Skipping 6 (task, backend) pairs:
[INFO]   Isaac-Velocity-Flat-Anymal-C-Direct-v0 × newton (seeds 42, 43, 44) — native_backend_mismatch (native: physx)
[INFO]   Isaac-NewtonOnly-v0 × physx (seeds 42, 43, 44) — preset_unsupported (available: [newton])
```

Final summary line at end of `run_dispatch` already aggregates by
reason via `skip_kinds = ", ".join(sorted({s.reason for s in state.skipped}))`,
so the new reason appears automatically.

## 8. Benchmark-script defense

`scripts/benchmarks/benchmark_{rsl_rl,skrl}.py` validate the requested
preset before injection (per the preset-handling fix). The new branch
sits between `has_physics_preset == False` and the existing
`preset_unsupported:` exit-2 path:

```python
if args_cli.backend is not None:
    existing_presets = [a for a in hydra_args if a.startswith("presets=")]
    if existing_presets:
        # ... existing operator-override path (warning + skip validation)
    else:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_tasks.utils.presets import has_physics_preset

        try:
            _raw_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        except Exception:
            # ... existing defensive fallback (warning + inject unchecked)
            ...
        else:
            if has_physics_preset(_raw_cfg, args_cli.backend):
                # Existing happy path: inject the named preset.
                hydra_args = [f"presets={args_cli.backend}"] + hydra_args
            elif _native_backend_matches(_raw_cfg, args_cli.backend):
                # NEW: task has no preset system; sim.physics type matches the
                # request → run on native backend without injection.
                print(
                    f"[INFO] task {args_cli.task!r} has no '{args_cli.backend}' "
                    f"preset; running on native {args_cli.backend} backend (no "
                    f"injection).",
                    file=sys.stderr,
                )
                # No injection — hydra_args unchanged.
            else:
                # Existing safety net: drift between yaml and runtime.
                sys.stderr.write(
                    f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
                    f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
                    f"re-enumerate {{physx,newton}}_envs.yaml.\n"
                )
                sys.exit(2)
```

`_native_backend_matches(raw_cfg, requested)` is colocated with the
existing logic and mirrors `_derive_native_backend` from §6:

```python
def _native_backend_matches(raw_cfg, requested: str) -> bool:
    sim = getattr(raw_cfg, "sim", None)
    if sim is None:
        return False
    physics = getattr(sim, "physics", None)
    if physics is None:
        return requested == "physx"
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_newton.physics import NewtonCfg
    try:
        from isaaclab_ovphysx.physics import OvPhysxCfg
    except ImportError:
        OvPhysxCfg = None
    if isinstance(physics, PhysxCfg):
        return requested == "physx"
    if isinstance(physics, NewtonCfg):
        return requested == "newton"
    if OvPhysxCfg is not None and isinstance(physics, OvPhysxCfg):
        return requested == "ovphysx"
    return False
```

Two intentional choices:

- **Helper duplicated between scripts**, mirroring the existing
  duplication of the preset-injection block. The shared-helper
  refactor is out of scope for this spec (same as the preset-handling
  spec).
- **`[INFO]` (not `[WARNING]`)** for the silent-injection-skip path.
  This is the *expected* outcome for native-only tasks — the queue
  filter already approved the (task, backend) pair as runnable.

The CLI test files (`scripts/benchmarks/tests/test_benchmark_*_cli.py`)
have local `_inject_preset_with_validation` mirrors that need
extending to take a `_native_backend_matches_fn` stub. Both production
and mirror change in lockstep, same as the preset-handling pattern.

## 9. Migration & rollout

Single PR, ordered commits:

1. **Schema additions** — `EnvEntry.native_backend` +
   `SkippedEntry.native_backend`, `_ENTRY_FIELD_ORDER` extension,
   `setdefault` for backward-compat, `_skipped_to_dict` /
   `_skipped_from_dict` updated, `state.py:SCHEMA_VERSION` bumped
   `"1.1"` → `"1.2"`. No behaviour change yet.
2. **`merge` propagation** — `merge` carries `native_backend` as a
   derived field on both branches.
3. **Enumerator integration** — `_derive_native_backend` helper +
   `build_entry_from_task_spec` calls it (with the
   `native_backend_fn=None` injection pattern).
4. **Yaml regeneration** — re-run both enumerators. One commit, only
   yaml diffs.
5. **Asgard queue rule 2** — `_expand_env_list` adds the
   `native_backend_mismatch` gate.
6. **Runner pre-dispatch summary** — group by `(task_id, backend, reason)`
   and show `native: X` for the new reason.
7. **Benchmark scripts** — both add `_native_backend_matches` +
   the new skip-injection branch.

No flag-gating, no migration window. Pre-fix yaml works because
`native_backend == None` falls through both rules harmlessly.

## 10. Test plan

### Unit tests

- **`tools/odin/tests/test_env_list.py`** (extend):
  - `test_roundtrip_preserves_native_backend` — yaml round-trip with
    `native_backend="physx"`.
  - `test_load_yaml_without_native_backend_defaults_to_none`.
  - `test_merge_refreshes_native_backend_on_existing_row`.
  - `test_merge_carries_native_backend_for_new_row`.
  - `test_build_entry_native_backend_*` — four cases: `None` (defaults
    to physx), `PhysxCfg()`, `NewtonCfg()`, `PresetCfg`-subclass.
    Stub `native_backend_fn`.

- **`tools/odin/tests/test_asgard_queue.py`** (extend):
  - `test_native_mismatch_skips_with_telemetry` — `presets_available=[]`,
    `native_backend="physx"`, request newton → 1 SkippedEntry per seed,
    `reason="native_backend_mismatch"`, `native_backend="physx"`.
  - `test_native_match_passes_through_to_runtime` — match → JobEntry
    created.
  - `test_unknown_native_passes_through` — `native_backend=None`
    → JobEntry created.
  - `test_preset_unsupported_takes_precedence_over_native` — pins
    rule ordering.

- **`tools/odin/tests/test_asgard_state.py`** (extend):
  - `test_roundtrip_skipped_with_native_backend` — round-trip a
    SkippedEntry with `native_backend="physx"`.
  - `test_read_skipped_without_native_backend_defaults_to_none`.
  - `test_schema_version_writes_1_2` — new dispatch.json files carry
    `schema_version == "1.2"`.
  - `test_read_v1_1_dispatch_json_loads_with_none_native_backend` —
    handcrafted 1.1 file loads cleanly; existing skipped[] entries
    surface with `native_backend=None`.

- **`tools/odin/tests/test_asgard_runner.py`** (extend):
  - `test_pre_dispatch_summary_renders_native_mismatch_line` — capsys
    asserts the stdout block has `native: physx`.

- **`scripts/benchmarks/tests/test_benchmark_{rsl_rl,skrl}_cli.py`**
  (extend each):
  - `test_validation_skips_injection_when_native_matches`.
  - `test_validation_still_blocks_when_native_mismatches`.

### Integration test

- **`tools/odin/tests/test_asgard_integration.py`** (extend):
  - `test_native_match_runs_unsupported_pair_routes_to_skipped` —
    yaml with one row `native_backend="physx"` + dispatch on newton →
    1 SkippedEntry with `reason="native_backend_mismatch"` in the
    on-disk `dispatch.json`.

### Aggregator test

- **`tools/odin/tests/test_valhalla_aggregator.py`** — none required
  (aggregator already passes `reason` through verbatim, pinned by the
  preset-handling spec).

Roughly **15-18 new tests** across the suite. Plus 1 yaml regeneration
commit and the 6 code commits from §9.

## 11. Risks & open questions

- **Yaml drift between enumeration and dispatch.** Same risk as the
  preset-handling fix. If `native_backend` becomes stale (e.g. task
  swaps from physx-native to newton-native upstream and yaml isn't
  re-enumerated), the queue filter approves a row that runtime
  rejects. The benchmark-script safety net catches this via the
  existing `preset_unsupported:` exit-2 path. Acceptable.

- **`PresetCfg` subclass with default == `PhysxCfg()`**.
  `_derive_native_backend` returns `None` for any `PresetCfg`-subclass
  instance because `presets_available` is supposed to be the source
  of truth for those rows. This means a `PresetCfg` subclass with
  `default=PhysxCfg(), newton=NewtonCfg()` (so
  `presets_available=[newton]`) defaults to physx but
  `presets_available` doesn't list physx as a named alternative.
  Today's behaviour: `--backend physx` → `preset_unsupported` (via
  rule 1's `presets_available` check). After this fix: still
  `preset_unsupported` (rule 1 still fires; rule 2 only fires when
  `presets_available=[]`). This is correct: a task that *has* a
  preset system but doesn't expose physx as a named option should
  not be silently rerouted to its default.

- **OvPhysx-native tasks under Odin.** These exist (Ant lists OvPhysx
  in its preset alternatives) but ovphysx isn't dispatched today.
  Tasks with `native_backend="ovphysx"` always fall into
  `native_backend_mismatch` against any Odin request. Acceptable
  pending ovphysx wiring.

- **Operator override path**. Both benchmark scripts let an explicit
  `presets=...` in `hydra_args` bypass validation. Same as the
  preset-handling fix; this spec doesn't change that. An operator who
  *wants* to force physx on a no-preset task can still pass
  `presets=physx` directly and Hydra will fail with its own error —
  but that's the operator's choice.

- **dispatch.json schema bump 1.1 → 1.2.** The major-match validator
  in `state.py` accepts any 1.x file, so prior 1.1 `dispatch.json`
  files keep loading on resume. The new optional `native_backend`
  field on SkippedEntry defaults to `None` on read via
  `_skipped_from_dict`'s defensive `.get(...)`. No coordinated
  reader/writer upgrade needed beyond bumping `SCHEMA_VERSION`.

## 12. Out of scope

- Dashboard rendering of `native_backend_mismatch` — T4.2.
- Sharing the `_native_backend_matches` helper between
  `benchmark_rsl_rl.py` and `benchmark_skrl.py` — separate refactor.
- OvPhysx dispatch wiring under Odin — separate task.
- Re-classifying historical `preset_unsupported` failures from prior
  dispatches — old `dispatch.json` files don't get rewritten.

## Summary

Stamp `native_backend: str | None` on every yaml row at enumeration
time; gate Asgard's queue filter on a second rule that catches
silent-swap requests with a distinct `native_backend_mismatch`
reason; teach the benchmark scripts to skip preset injection when
the cfg type matches the request. Native-only tasks (Anymal-C Flat,
Quadcopter, etc.) now run cleanly on their native backend; mismatched
requests get clean telemetry. Six code commits, one yaml regen,
`dispatch.json` schema bumps 1.1 → 1.2 (additive — major-match
validator already accepts 1.x). No behaviour regressions on yaml
that hasn't been re-stamped yet.
