# Odin Preset-Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop Odin from dispatching `(task, backend)` pairs the task doesn't actually support; route them into a new `dispatch.json` `skipped[]` array; add a fail-fast safety net at the benchmark-script layer keyed on `preset_unsupported:`.

**Architecture:** Yaml is the source of truth — `EnvEntry.presets_available: list[str]` populated by the enumerator. Asgard's `_expand_env_list` filters by it; filtered pairs land in a parallel `SkippedEntry` list that flows through `dispatch.json`. `benchmark_{rsl_rl,skrl}.py` validates the requested preset before injection and exits with a stable stderr prefix; `worker._classify` maps that prefix to `failure_kind="preset_unsupported"`.

**Tech Stack:** Python 3.10+, dataclasses, PyYAML, pytest. Touches `tools/odin/common/env_list.py`, `tools/odin/asgard/{jobs,state,runner,worker}.py`, `scripts/benchmarks/benchmark_{rsl_rl,skrl}.py`, plus tests under `tools/odin/tests/` and `scripts/benchmarks/tests/`.

---

## File map (locked decomposition)

**Modify:**
- `tools/odin/common/env_list.py` — add `presets_available: list[str]` to `EnvEntry`, extend `_ENTRY_FIELD_ORDER`, extend `_entry_from_dict` with a `setdefault`, propagate the field through `merge`, add a `raw_cfg_loader` injection point on `build_entry_from_task_spec` and call it.
- `tools/odin/asgard/jobs.py` — add `SkippedEntry` dataclass, extend `FailureInfo.kind` docstring, change `_expand_env_list` return type to `tuple[list[JobEntry], list[SkippedEntry]]`, change `build_queue_from_env_lists` return type identically.
- `tools/odin/asgard/state.py` — bump `SCHEMA_VERSION` to `"1.1"`; switch validator from strict-equality to major-match; serialize/deserialize `skipped[]` round-trip.
- `tools/odin/asgard/runner.py` — thread the new `(jobs, skipped)` tuple through `run_dispatch`, store `skipped` on `DispatchState`, emit pre-dispatch stdout block, extend final summary line.
- `tools/odin/asgard/worker.py` — add `preset_unsupported:` branch to `_classify`.
- `scripts/benchmarks/benchmark_rsl_rl.py` — gate the existing `presets=` injection on `has_physics_preset(raw_cfg, args_cli.backend)`.
- `scripts/benchmarks/benchmark_skrl.py` — same change, mirrored.

**Test files to extend (no new files):**
- `tools/odin/tests/test_env_list.py`
- `tools/odin/tests/test_asgard_queue.py`
- `tools/odin/tests/test_asgard_state.py`
- `tools/odin/tests/test_asgard_runner.py`
- `tools/odin/tests/test_asgard_worker.py`
- `tools/odin/tests/test_asgard_integration.py`
- `tools/odin/tests/test_valhalla_aggregator.py`
- `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`
- `scripts/benchmarks/tests/test_benchmark_skrl_cli.py`

**Data files to refresh (one commit, generated):**
- `tools/odin/config/physx_envs.yaml`
- `tools/odin/config/newton_envs.yaml`
- `tools/odin/config/newton_gap_candidates.yaml` *(if the newton enumerator regenerates it)*

---

## Task 1: `EnvEntry.presets_available` field + serializer plumbing

**Files:**
- Modify: `tools/odin/common/env_list.py:150-194` (`_ENTRY_FIELD_ORDER` + `EnvEntry`); `tools/odin/common/env_list.py:209-239` (`_entry_from_dict`)
- Test: `tools/odin/tests/test_env_list.py` (extend at end of file, before any new section)

- [ ] **Step 1: Write the failing test for round-trip with the new field**

Append to `tools/odin/tests/test_env_list.py`:

```python
def test_roundtrip_preserves_presets_available(tmp_path: Path):
    """presets_available list survives load + dump."""
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            presets_available=["physx", "newton"],
        )
    ]
    out = tmp_path / "envs.yaml"
    write_env_list(out, el, generator="test")
    reloaded = load_env_list(out)
    assert reloaded.groups["direct/ant"][0].presets_available == ["physx", "newton"]


def test_load_yaml_without_presets_available_defaults_to_empty(tmp_path: Path):
    """Backward-compat: pre-1.0 yaml that doesn't carry the field reads as []."""
    yaml_text = """\
schema_version: '1.0'
generator: legacy
groups:
  direct/ant:
    - task_id: Isaac-Ant-Direct-v0
      entry_point: ep:E
      env_cfg_entry_point: ec:E
      group: direct/ant
      has_rsl_rl: true
      has_skrl: true
      has_rl_games: false
      framework: rsl_rl
      num_envs: 4096
      max_iterations: 300
      keep: true
      status: current
      suspected_gap: null
      notes: ''
"""
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_text)
    el = load_env_list(p)
    assert el.groups["direct/ant"][0].presets_available == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_env_list.py::test_roundtrip_preserves_presets_available tools/odin/tests/test_env_list.py::test_load_yaml_without_presets_available_defaults_to_empty -v --confcutdir=tools/odin`

Expected: FAIL — `EnvEntry.__init__() got an unexpected keyword argument 'presets_available'` and a follow-up dump-shape mismatch.

- [ ] **Step 3: Add the field to the dataclass and serializer ordering**

Edit `tools/odin/common/env_list.py`. Replace `_ENTRY_FIELD_ORDER` (lines 150-165) with:

```python
_ENTRY_FIELD_ORDER = [
    "task_id",
    "entry_point",
    "env_cfg_entry_point",
    "group",
    "has_rsl_rl",
    "has_skrl",
    "has_rl_games",
    "framework",
    "num_envs",
    "max_iterations",
    "keep",
    "status",
    "suspected_gap",
    "presets_available",
    "notes",
]
```

In the `EnvEntry` dataclass (lines 168-194), add the field immediately before `notes`:

```python
    presets_available: list[str] = field(default_factory=list)
    notes: str = ""
```

In `_entry_from_dict` (lines 209-239), add the missing-field default. Insert this line in the `setdefault` block, e.g. right after `known.setdefault("status", "current")`:

```python
    known.setdefault("presets_available", [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin`

Expected: PASS — including the two new tests and all existing 30+ tests.

- [ ] **Step 5: Commit**

```bash
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "EnvEntry: add presets_available list[str] field

Stores per-task preset support (e.g. [\"physx\", \"newton\"]) so the
Asgard queue builder can filter unsupported (task, backend) pairs
before dispatch.  Defaults to an empty list, which the filter treats
as 'unknown' (pass through to the runtime safety net) so existing
yaml that hasn't been re-enumerated continues to work unchanged."
```

---

## Task 2: `merge` propagates `presets_available` as a derived field

**Files:**
- Modify: `tools/odin/common/env_list.py:312-402` (`merge`)
- Test: `tools/odin/tests/test_env_list.py` (extend after the existing `test_merge_refreshes_derived_fields` block at ~line 211)

- [ ] **Step 1: Write the failing tests**

Append to `tools/odin/tests/test_env_list.py` near the other `test_merge_*` tests:

```python
def test_merge_refreshes_presets_available_on_existing_row():
    """presets_available is derived from runtime introspection — refresh it."""
    old = EnvEntry(
        task_id="Isaac-Ant-Direct-v0",
        entry_point="ep:E",
        env_cfg_entry_point="ec:E",
        group="direct/ant",
        has_rsl_rl=True,
        has_skrl=True,
        framework="rsl_rl",
        num_envs=4096,
        max_iterations=300,
        keep=True,
        presets_available=["physx"],  # stale: task gained newton support
    )
    new = EnvEntry(
        task_id="Isaac-Ant-Direct-v0",
        entry_point="ep:E",
        env_cfg_entry_point="ec:E",
        group="direct/ant",
        has_rsl_rl=True,
        has_skrl=True,
        framework="rsl_rl",
        num_envs=4096,
        max_iterations=300,
        keep=True,
        presets_available=["physx", "newton"],
    )
    existing = EnvList()
    existing.groups["direct/ant"] = [old]
    merged = merge(existing, [new])
    row = merged.groups["direct/ant"][0]
    assert row.presets_available == ["physx", "newton"]


def test_merge_carries_presets_available_for_new_row():
    new = EnvEntry(
        task_id="Isaac-NewTask-Direct-v0",
        entry_point="ep:E",
        env_cfg_entry_point="ec:E",
        group="direct/newtask",
        has_rsl_rl=True,
        has_skrl=False,
        framework="rsl_rl",
        num_envs=1024,
        max_iterations=100,
        keep=True,
        presets_available=["newton"],
    )
    merged = merge(EnvList(), [new])
    row = merged.groups["direct/newtask"][0]
    assert row.status == "new"
    assert row.presets_available == ["newton"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_env_list.py::test_merge_refreshes_presets_available_on_existing_row tools/odin/tests/test_env_list.py::test_merge_carries_presets_available_for_new_row -v --confcutdir=tools/odin`

Expected: FAIL — both tests assert on a field that the merged row constructor doesn't populate, so they get the dataclass default `[]`.

- [ ] **Step 3: Update `merge` to propagate the field**

In `tools/odin/common/env_list.py`, find the two `EnvEntry(...)` constructions inside `merge` (lines ~358-373 for the "new only" branch and lines ~375-392 for the "old + new" branch).

For the "new only" branch (line ~358), add `presets_available=new.presets_available,` after `suspected_gap=new.suspected_gap,`:

```python
            merged_entry = EnvEntry(
                task_id=new.task_id,
                entry_point=new.entry_point,
                env_cfg_entry_point=new.env_cfg_entry_point,
                group=new.group,
                has_rsl_rl=new.has_rsl_rl,
                has_skrl=new.has_skrl,
                has_rl_games=new.has_rl_games,
                framework=new.framework,
                num_envs=new.num_envs,
                max_iterations=new.max_iterations,
                keep=new.keep,
                status="new",
                notes=new.notes,
                suspected_gap=new.suspected_gap,
                presets_available=new.presets_available,
            )
```

For the "old + new" branch (line ~375), add `presets_available=new.presets_available,` after `status="current",` — `presets_available` is a derived field, so it takes the fresh value:

```python
            merged_entry = EnvEntry(
                task_id=new.task_id,
                # Derived / refreshed from current registry:
                entry_point=new.entry_point,
                env_cfg_entry_point=new.env_cfg_entry_point,
                group=new.group,
                has_rsl_rl=new.has_rsl_rl,
                has_skrl=new.has_skrl,
                has_rl_games=new.has_rl_games,
                presets_available=new.presets_available,
                # Preserved from user edits:
                framework=old.framework,
                num_envs=old.num_envs,
                max_iterations=old.max_iterations,
                keep=old.keep,
                notes=old.notes,
                suspected_gap=old.suspected_gap,
                status="current",
            )
```

(The "stale" branch at line ~399 uses `**asdict(old)`, which automatically carries the field — no edit needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin`

Expected: PASS for all `test_merge_*` tests, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "env_list.merge: refresh presets_available on every pass

presets_available is derived from runtime introspection (the same way
entry_point and has_rsl_rl are), so it must be refreshed on each
enumeration, not preserved from the prior yaml.  The 'old + new'
branch takes new.presets_available; the 'new only' branch carries
the discovered list verbatim; the 'stale' branch is unchanged
because it uses asdict(old) which already carries the field."
```

---

## Task 3: `build_entry_from_task_spec` populates `presets_available`

**Files:**
- Modify: `tools/odin/common/env_list.py:501-575` (`build_entry_from_task_spec`)
- Test: `tools/odin/tests/test_env_list.py` (extend after `test_build_entry_skrl_only` at ~line 381)

The builder gains a `raw_cfg_loader` injection (mirrors the existing `defaults_loader` injection) so tests can stub out `has_physics_preset`. Real callers get the default loader that imports from isaaclab.

- [ ] **Step 1: Write failing tests for the four cases**

Append to `tools/odin/tests/test_env_list.py` near the existing `test_build_entry_*` tests:

```python
class _StubRawCfg:
    """Sentinel object passed to the stub has_physics_preset."""

    def __init__(self, supports: set[str]):
        self.supports = supports


def _stub_raw_cfg_loader(supports: set[str]):
    def _loader(task_id: str):
        return _StubRawCfg(supports)
    return _loader


def _stub_has_physics_preset(raw_cfg, name: str) -> bool:
    return name in raw_cfg.supports


def test_build_entry_presets_available_both():
    spec = _FakeSpec(
        id="Isaac-Dual-v0",
        entry_point="isaaclab_tasks.direct.dual:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "isaaclab_tasks.direct.dual.cfg:Cfg",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=_stub_raw_cfg_loader({"physx", "newton"}),
        has_physics_preset_fn=_stub_has_physics_preset,
    )
    assert e.presets_available == ["physx", "newton"]


def test_build_entry_presets_available_physx_only():
    spec = _FakeSpec(
        id="Isaac-PhysxOnly-v0",
        entry_point="isaaclab_tasks.direct.po:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "isaaclab_tasks.direct.po.cfg:Cfg",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=_stub_raw_cfg_loader({"physx"}),
        has_physics_preset_fn=_stub_has_physics_preset,
    )
    assert e.presets_available == ["physx"]


def test_build_entry_presets_available_none():
    spec = _FakeSpec(
        id="Isaac-NoPresets-v0",
        entry_point="isaaclab_tasks.direct.np:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "isaaclab_tasks.direct.np.cfg:Cfg",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=_stub_raw_cfg_loader(set()),
        has_physics_preset_fn=_stub_has_physics_preset,
    )
    assert e.presets_available == []


def test_build_entry_skips_preset_query_when_loader_raises():
    """A failure to load raw_cfg leaves presets_available empty (silent fall-through)."""

    def _raises(task_id: str):
        raise RuntimeError("load failed")

    spec = _FakeSpec(
        id="Isaac-Crashy-v0",
        entry_point="isaaclab_tasks.direct.cr:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "isaaclab_tasks.direct.cr.cfg:Cfg",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=_raises,
        has_physics_preset_fn=_stub_has_physics_preset,
    )
    assert e.presets_available == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_env_list.py -k presets_available -v --confcutdir=tools/odin`

Expected: FAIL — `build_entry_from_task_spec` doesn't yet take `raw_cfg_loader` / `has_physics_preset_fn` kwargs.

- [ ] **Step 3: Extend `build_entry_from_task_spec`**

In `tools/odin/common/env_list.py`, find `build_entry_from_task_spec` (line ~501).

Add a default raw-cfg loader near the top of the file, right after `_ISAACLAB_TASKS_PREFIX = "isaaclab_tasks."`:

```python
def _default_raw_cfg_loader(task_id: str):
    """Default raw-cfg loader for ``build_entry_from_task_spec``.

    Deferred import — isaaclab_tasks must be importable, which requires
    the Kit app to be up. Caller's responsibility (matches the contract
    of :func:`load_shipped_training_defaults`).
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    return load_cfg_from_registry(task_id, "env_cfg_entry_point")
```

Then update `build_entry_from_task_spec`'s signature and body. Replace the function (lines ~501-575) with:

```python
def build_entry_from_task_spec(
    task_spec: Any,
    *,
    defaults_loader=load_shipped_training_defaults,
    raw_cfg_loader=_default_raw_cfg_loader,
    has_physics_preset_fn=None,
) -> EnvEntry:
    """Construct an :class:`EnvEntry` from a gym ``EnvSpec``-like object.

    Args:
        task_spec: An object with ``id``, ``entry_point``, and ``kwargs``
            attributes (gymnasium's ``EnvSpec`` satisfies this).
        defaults_loader: Callable taking ``(task_id, framework)`` and returning
            ``(num_envs, max_iterations)``. Defaults to the real
            :func:`load_shipped_training_defaults`; tests pass a stub.
        raw_cfg_loader: Callable taking ``task_id`` and returning the task's
            raw env cfg (the unresolved-PresetCfg form expected by
            :func:`~tools.odin.common.presets.has_physics_preset`). Defaults
            to a thin wrapper around ``load_cfg_from_registry``; tests pass
            a stub. A loader that raises results in
            ``presets_available=[]`` (the row falls through to the runtime
            safety net).
        has_physics_preset_fn: Callable matching
            :func:`~tools.odin.common.presets.has_physics_preset`'s signature.
            Defaults to the real function; tests pass a stub.

    Returns:
        A freshly-built :class:`EnvEntry` with ``status="current"`` and
        ``presets_available`` populated (empty list if no preset query
        was possible).
    """
    if has_physics_preset_fn is None:
        from tools.odin.common.presets import has_physics_preset as _real_hpp

        has_physics_preset_fn = _real_hpp

    kwargs = task_spec.kwargs or {}
    has_rsl_rl = "rsl_rl_cfg_entry_point" in kwargs
    has_skrl = "skrl_cfg_entry_point" in kwargs
    has_rl_games = "rl_games_cfg_entry_point" in kwargs
    framework = suggest_framework(has_rsl_rl, has_skrl)
    env_cfg_ep = kwargs.get("env_cfg_entry_point") or ""
    group_from_env_cfg = derive_group(env_cfg_ep) if isinstance(env_cfg_ep, str) else "unknown"
    group_from_entry = derive_group(task_spec.entry_point or "")
    group = group_from_env_cfg if group_from_env_cfg != "unknown" else group_from_entry

    num_envs: int | None = None
    max_iterations: int | None = None
    notes = ""

    if framework is None:
        if has_rl_games:
            notes = (
                "rl_games-only registration — not dispatched by Odin. Migrate to rsl_rl or skrl to enable benchmarking."
            )
        else:
            notes = "No rsl_rl or skrl entry point registered."
    else:
        try:
            num_envs, max_iterations = defaults_loader(task_spec.id, framework)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING env_list: defaults_loader raised for {task_spec.id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            num_envs = None
            max_iterations = None
        if num_envs is None or max_iterations is None:
            notes = (
                "Could not resolve training defaults from shipped framework cfg; "
                "review the cfg and fill num_envs / max_iterations manually."
            )

    keep = framework is not None and num_envs is not None and max_iterations is not None

    presets_available: list[str] = []
    try:
        raw_cfg = raw_cfg_loader(task_spec.id)
    except Exception as exc:  # noqa: BLE001 — preset query never aborts row construction
        print(
            f"WARNING env_list: raw_cfg_loader raised for {task_spec.id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raw_cfg = None
    if raw_cfg is not None:
        for name in ("physx", "newton"):
            try:
                if has_physics_preset_fn(raw_cfg, name):
                    presets_available.append(name)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING env_list: has_physics_preset raised for {task_spec.id} / {name}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    return EnvEntry(
        task_id=task_spec.id,
        entry_point=task_spec.entry_point or "",
        env_cfg_entry_point=kwargs.get("env_cfg_entry_point"),
        group=group,
        has_rsl_rl=has_rsl_rl,
        has_skrl=has_skrl,
        has_rl_games=has_rl_games,
        framework=framework,
        num_envs=num_envs,
        max_iterations=max_iterations,
        keep=keep,
        status="current",
        notes=notes,
        suspected_gap=None,
        presets_available=presets_available,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin`

Expected: PASS — all four new tests, plus the existing `test_build_entry_*` tests (the new defaults are backward-compatible because the new kwargs have defaults).

- [ ] **Step 5: Commit**

```bash
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "env_list: build_entry_from_task_spec stamps presets_available

Calls has_physics_preset(raw_cfg, name) for both 'physx' and 'newton'
and stamps the resulting list onto the EnvEntry.  Two new injection
points (raw_cfg_loader, has_physics_preset_fn) keep the function
unit-testable without isaaclab; defaults route through the existing
load_cfg_from_registry + tools.odin.common.presets path used by
classify_for_newton.  Loader / preset-query exceptions degrade to
presets_available=[], which the runtime safety net catches."
```

---

## Task 4: Re-enumerate yaml files

**Files:**
- Modify (generated): `tools/odin/config/physx_envs.yaml`, `tools/odin/config/newton_envs.yaml`, possibly `tools/odin/config/newton_gap_candidates.yaml`

This task contains no test code — it's a single regeneration commit that materializes `presets_available` on every existing row. Run after Task 3 has landed.

- [ ] **Step 1: Re-run the physx enumerator**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
```

Expected: Script completes; `tools/odin/config/physx_envs.yaml` is rewritten in place. The merge logic preserves `keep`, `notes`, `status`, `suspected_gap`. New `presets_available` keys appear on every row.

- [ ] **Step 2: Re-run the newton enumerator**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

Expected: Script completes; `tools/odin/config/newton_envs.yaml` and `tools/odin/config/newton_gap_candidates.yaml` are rewritten. New `presets_available` keys present.

- [ ] **Step 3: Sanity-check the diff**

```bash
git diff tools/odin/config/*.yaml | head -80
```

Expected: every changed row gains a `presets_available: [...]` line. The two known-broken rows (`Isaac-Velocity-Flat-Anymal-C-Direct-v0` and `Isaac-Quadcopter-Direct-v0` in `physx_envs.yaml`) should land with `presets_available: []`. Tasks with both backends (e.g. Ant) should show `presets_available: [physx, newton]` (order matters: physx first per the for-loop in Task 3).

- [ ] **Step 4: Spot-check Anymal-C and Quadcopter rows**

```bash
python3 -c "
import yaml
data = yaml.safe_load(open('tools/odin/config/physx_envs.yaml'))
for grp in data['groups'].values():
    for r in grp:
        if r['task_id'] in ('Isaac-Velocity-Flat-Anymal-C-Direct-v0', 'Isaac-Quadcopter-Direct-v0'):
            print(r['task_id'], '->', r['presets_available'])
"
```

Expected output:

```
Isaac-Velocity-Flat-Anymal-C-Direct-v0 -> []
Isaac-Quadcopter-Direct-v0 -> []
```

If either prints something other than `[]`, stop — the task does have a preset and Task 3's loader is producing a wrong answer; investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add tools/odin/config/physx_envs.yaml tools/odin/config/newton_envs.yaml tools/odin/config/newton_gap_candidates.yaml
git commit -m "Re-enumerate env-list yamls with presets_available

One-shot regeneration after build_entry_from_task_spec learned to
stamp presets_available.  Anymal-C Flat and Quadcopter (both
pre-preset-system) land with presets_available: [].  Dual-preset
tasks (Ant, Cartpole, Humanoid, etc.) land with [physx, newton].
The Asgard filter (next task) is what actually consumes this."
```

---

## Task 5: `SkippedEntry` dataclass + `_expand_env_list` filter

**Files:**
- Modify: `tools/odin/asgard/jobs.py:16-23` (FailureInfo docstring), `tools/odin/asgard/jobs.py:64-95` (`_expand_env_list`), `tools/odin/asgard/jobs.py:98-134` (`build_queue_from_env_lists`)
- Test: `tools/odin/tests/test_asgard_queue.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_asgard_queue.py`. First update `_env` to take an explicit `presets_available` arg:

```python
def _env(
    task_id: str,
    framework: str = "rsl_rl",
    keep: bool = True,
    status: str = "current",
    presets_available: list[str] | None = None,
) -> EnvEntry:
    return EnvEntry(
        task_id=task_id,
        entry_point="isaaclab_tasks.direct.ant:AntEnv",
        env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
        group="direct/ant",
        has_rsl_rl=True,
        has_skrl=True,
        has_rl_games=False,
        framework=framework,
        num_envs=4096,
        max_iterations=300,
        keep=keep,
        status=status,
        presets_available=list(presets_available) if presets_available is not None else [],
    )
```

Then append the new tests:

```python
def test_supported_pair_produces_jobs(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0", presets_available=["physx", "newton"])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42, 43, 44],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 3
    assert skipped == []


def test_unsupported_pair_skips_with_telemetry(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Velocity-Flat-Anymal-C-Direct-v0", presets_available=[])],
    )
    # Empty list = unknown → pass through (not skipped). Use [newton] to
    # exercise the actual skip path against the physx backend.
    physx2 = _write_env_list(
        tmp_path,
        "physx2.yaml",
        [_env("Isaac-NewtonOnly-v0", presets_available=["newton"])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx2,
        newton_yaml=None,
        seeds=[42, 43, 44],
        dispatch_id="20260427-100000",
    )
    assert jobs == []
    assert len(skipped) == 3
    s = skipped[0]
    assert s.task_id == "Isaac-NewtonOnly-v0"
    assert s.framework == "rsl_rl"
    assert s.backend == "physx"
    assert s.seed in {42, 43, 44}
    assert s.reason == "preset_unsupported"
    assert s.presets_available == ["newton"]


def test_empty_presets_available_passes_through(tmp_path: Path):
    """Unknown preset support (empty list) must NOT trigger the skip path."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Unknown-Presets-v0", presets_available=[])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 1
    assert skipped == []


def test_dual_preset_supports_both_backends(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Dual-v0", presets_available=["physx", "newton"])],
    )
    newton = _write_env_list(
        tmp_path,
        "newton.yaml",
        [_env("Isaac-Dual-v0", presets_available=["physx", "newton"])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=newton,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 2
    assert {j.backend for j in jobs} == {"physx", "newton"}
    assert skipped == []


def test_include_filter_runs_before_preset_filter(tmp_path: Path):
    """Rows excluded by --include must NOT appear in skipped[]."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [
            _env("Isaac-Ant-Direct-v0", presets_available=["physx"]),
            _env("Isaac-NewtonOnly-v0", presets_available=["newton"]),
        ],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
        include_filter=["Isaac-Ant-*"],
    )
    assert len(jobs) == 1
    assert jobs[0].task_id == "Isaac-Ant-Direct-v0"
    assert skipped == []  # NewtonOnly filtered out before preset gate
```

Also update existing tests in this file that destructure `build_queue_from_env_lists`'s return — find every `jobs = build_queue_from_env_lists(...)` and replace with `jobs, _ = build_queue_from_env_lists(...)`. The seven existing tests (`test_expand_one_row_one_seed`, `test_expand_multiple_seeds`, `test_combines_physx_and_newton`, `test_skips_keep_false_rows`, `test_skips_stale_rows`, `test_include_filter_fnmatch`, `test_empty_seeds_raises`) all need this update.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin`

Expected: FAIL — `build_queue_from_env_lists` returns a `list`, not a tuple.

- [ ] **Step 3: Update `tools/odin/asgard/jobs.py`**

Replace the entire `tools/odin/asgard/jobs.py` `__all__` line and the `FailureInfo` docstring + add `SkippedEntry` immediately after `FailureInfo`. Also update `_expand_env_list` and `build_queue_from_env_lists` signatures.

Concretely:

Update `__all__` (line 16):

```python
__all__ = ["JobEntry", "FailureInfo", "SkippedEntry", "build_queue_from_env_lists"]
```

Update `FailureInfo`'s docstring (lines 19-25) to include the new kind:

```python
@dataclass
class FailureInfo:
    """Classified failure attached to a :class:`JobEntry` when ``status == 'failed'``.

    ``kind`` values:

    - ``infrastructure``: docker / SSH transport failure (retried).
    - ``hugin_crash``: training process exited non-zero with no
      Odin-recognised stderr signal.
    - ``hugin_malformed_bundle``: SSH succeeded, rsync pulled, but the
      bundle's manifest is missing or invalid.
    - ``timeout``: SSH wall-clock timeout fired.
    - ``preset_unsupported``: training process exited non-zero with a
      stderr line beginning ``preset_unsupported:`` — the requested
      preset doesn't exist for the task. Caught by the runtime safety
      net when yaml-stamped ``presets_available`` is stale.
    """

    kind: str
    message: str
    details: dict[str, object] = field(default_factory=dict)
```

Add `SkippedEntry` after `FailureInfo` (insert immediately after the `FailureInfo` block, before `JobEntry`):

```python
@dataclass
class SkippedEntry:
    """An (task, framework, backend, seed) pair that the queue builder rejected.

    Lives next to :class:`JobEntry` because both are persisted into
    ``dispatch.json`` (jobs[] and skipped[] respectively).  The current
    only ``reason`` is ``"preset_unsupported"``, but the type is open
    to additional reasons (e.g. future ``"capability_mismatch"``).
    """

    task_id: str
    framework: str
    backend: str
    seed: int
    reason: str
    presets_available: list[str] = field(default_factory=list)
```

Replace `_expand_env_list` (lines 64-95) with:

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
            # Preset-support gate. Empty list = unknown → pass through;
            # populated list with backend missing → skip.
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

Replace `build_queue_from_env_lists` (lines 98-134) — only the return-type and the body changes; docstring updates accordingly:

```python
def build_queue_from_env_lists(
    physx_yaml: Path | None,
    newton_yaml: Path | None,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None = None,
) -> tuple[list[JobEntry], list[SkippedEntry]]:
    """Expand curated env YAMLs across seeds into a flat ``(jobs, skipped)`` pair.

    Args:
        physx_yaml: Path to ``physx_envs.yaml`` (T2.1); ``None`` to skip PhysX.
        newton_yaml: Path to ``newton_envs.yaml`` (T2.1); ``None`` to skip Newton.
        seeds: Seeds to expand each kept row across. Must be non-empty.
        dispatch_id: UTC timestamp (``YYYYMMDD-HHMMSS``) shared by all run_ids
            in this dispatch.
        include_filter: Optional list of fnmatch patterns on ``task_id``; a row
            must match at least one pattern to be queued. Unset = keep all.

    Returns:
        ``(jobs, skipped)``. ``jobs`` is a list of :class:`JobEntry` rows in
        insertion order (PhysX first, then Newton). ``skipped`` is the list
        of :class:`SkippedEntry` rows for ``(task, backend, seed)`` triples
        that didn't match the row's ``presets_available`` — each
        ``--include``-passing seed of an unsupported task contributes one
        ``SkippedEntry``.

    Raises:
        ValueError: If neither YAML is provided or seeds is empty.
    """
    if physx_yaml is None and newton_yaml is None:
        raise ValueError("build_queue_from_env_lists needs at least one env list (physx_yaml or newton_yaml)")
    if not seeds:
        raise ValueError("build_queue_from_env_lists needs a non-empty seed list")

    jobs: list[JobEntry] = []
    skipped: list[SkippedEntry] = []
    if physx_yaml is not None:
        j, s = _expand_env_list(physx_yaml, "physx", seeds, dispatch_id, include_filter)
        jobs.extend(j)
        skipped.extend(s)
    if newton_yaml is not None:
        j, s = _expand_env_list(newton_yaml, "newton", seeds, dispatch_id, include_filter)
        jobs.extend(j)
        skipped.extend(s)
    return jobs, skipped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin`

Expected: PASS for all tests (existing 8 + 5 new = 13). Note: Task 6 (next) updates the runner.py callers, so cross-module breakage shows up there.

- [ ] **Step 5: Commit**

```bash
git add tools/odin/asgard/jobs.py tools/odin/tests/test_asgard_queue.py
git commit -m "Asgard: SkippedEntry + (task, backend) preset filter

_expand_env_list and build_queue_from_env_lists now return a
(jobs, skipped) tuple.  Rows with a populated presets_available
that doesn't include the requested backend produce one SkippedEntry
per seed instead of a JobEntry; rows with presets_available=[]
(unknown) pass through to the runtime safety net unchanged.
FailureInfo gains a documented 'preset_unsupported' kind for the
worker classifier (separate commit)."
```

---

## Task 6: Plumb `skipped[]` through `DispatchState` (state.py + runner.py)

**Files:**
- Modify: `tools/odin/asgard/state.py:34` (SCHEMA_VERSION), `tools/odin/asgard/state.py:48-60` (DispatchState), `tools/odin/asgard/state.py:120-162` (state-dict round trip), `tools/odin/asgard/state.py:141-144` (validator)
- Modify: `tools/odin/asgard/runner.py` — every `DispatchState(...)` constructor (~5 sites), the `build_queue_from_env_lists` call, and the final summary
- Test: `tools/odin/tests/test_asgard_state.py` (extend), `tools/odin/tests/test_asgard_runner.py` (extend)

- [ ] **Step 1: Write failing tests for state round-trip**

Append to `tools/odin/tests/test_asgard_state.py`:

```python
from tools.odin.asgard.jobs import SkippedEntry


def _state_with_skipped(jobs: list[JobEntry], skipped: list[SkippedEntry]) -> DispatchState:
    return DispatchState(
        schema_version="1.1",
        dispatch_id="20260427-100000",
        started_at="2026-04-27T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc123",
        fleet=[FleetSnapshot(host="h1", status="idle", current_run_id=None, last_error=None)],
        jobs=jobs,
        skipped=skipped,
    )


def test_roundtrip_skipped_array(tmp_path: Path):
    skipped = [
        SkippedEntry(
            task_id="Isaac-Velocity-Flat-Anymal-C-Direct-v0",
            framework="rsl_rl",
            backend="physx",
            seed=42,
            reason="preset_unsupported",
            presets_available=[],
        ),
        SkippedEntry(
            task_id="Isaac-NewtonOnly-v0",
            framework="rsl_rl",
            backend="physx",
            seed=43,
            reason="preset_unsupported",
            presets_available=["newton"],
        ),
    ]
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], skipped))
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded.schema_version == "1.1"
    assert len(reloaded.skipped) == 2
    s0 = reloaded.skipped[0]
    assert s0.task_id == "Isaac-Velocity-Flat-Anymal-C-Direct-v0"
    assert s0.reason == "preset_unsupported"
    assert s0.presets_available == []
    assert reloaded.skipped[1].presets_available == ["newton"]


def test_read_v1_0_dispatch_json_loads_with_empty_skipped(tmp_path: Path):
    """Reading a 1.0 file with no skipped key returns DispatchState.skipped == []."""
    path = tmp_path / "dispatch.json"
    path.write_text(
        '{"schema_version": "1.0", "dispatch_id": "old", '
        '"started_at": "2026-01-01T00:00:00Z", "ended_at": null, '
        '"seeds": [42], "commit_sha": "", "fleet": [], "jobs": []}'
    )
    s = read_dispatch_state(tmp_path)
    assert s is not None
    assert s.schema_version == "1.0"
    assert s.skipped == []


def test_read_rejects_major_version_2(tmp_path: Path):
    """Major-version mismatch is still a hard error."""
    path = tmp_path / "dispatch.json"
    path.write_text(
        '{"schema_version": "2.0", "dispatch_id": "future", '
        '"started_at": "x", "ended_at": null, "seeds": [], "commit_sha": "", '
        '"fleet": [], "jobs": []}'
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_dispatch_state(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin`

Expected: FAIL — `DispatchState` has no `skipped` field; the validator does strict equality against `"1.0"`.

- [ ] **Step 3: Update `tools/odin/asgard/state.py`**

Bump version (line 34):

```python
SCHEMA_VERSION = "1.1"
```

Update the import (line 22) to bring `SkippedEntry`:

```python
from tools.odin.asgard.jobs import FailureInfo, JobEntry, SkippedEntry
```

Add `skipped` field to `DispatchState` (line 48-60). Replace with:

```python
@dataclass
class DispatchState:
    """Complete on-disk state for one dispatch."""

    schema_version: str
    dispatch_id: str
    started_at: str  # UTC ISO-8601
    ended_at: str | None
    seeds: list[int]
    commit_sha: str
    fleet: list[FleetSnapshot]
    jobs: list[JobEntry]
    skipped: list[SkippedEntry] = field(default_factory=list)
```

Add `field` to the imports at the top of the file:

```python
from dataclasses import dataclass, field
```

Add a `_skipped_to_dict` / `_skipped_from_dict` pair right after `_job_from_dict` (~line 117):

```python
def _skipped_to_dict(s: SkippedEntry) -> dict[str, Any]:
    return {
        "task_id": s.task_id,
        "framework": s.framework,
        "backend": s.backend,
        "seed": s.seed,
        "reason": s.reason,
        "presets_available": list(s.presets_available),
    }


def _skipped_from_dict(d: dict[str, Any]) -> SkippedEntry:
    return SkippedEntry(
        task_id=str(d["task_id"]),
        framework=str(d["framework"]),
        backend=str(d["backend"]),
        seed=int(d["seed"]),
        reason=str(d.get("reason", "preset_unsupported")),
        presets_available=list(d.get("presets_available") or []),
    )
```

Update `_state_to_dict` (line 120) to emit `skipped`:

```python
def _state_to_dict(s: DispatchState) -> dict[str, Any]:
    return {
        "schema_version": s.schema_version,
        "dispatch_id": s.dispatch_id,
        "started_at": s.started_at,
        "ended_at": s.ended_at,
        "seeds": list(s.seeds),
        "commit_sha": s.commit_sha,
        "fleet": [
            {
                "host": f.host,
                "status": f.status,
                "current_run_id": f.current_run_id,
                "last_error": f.last_error,
            }
            for f in s.fleet
        ],
        "jobs": [_job_to_dict(j) for j in s.jobs],
        "skipped": [_skipped_to_dict(sk) for sk in s.skipped],
    }
```

Update `_state_from_dict` (line 141-162):

```python
def _state_from_dict(d: dict[str, Any]) -> DispatchState:
    got_schema = str(d.get("schema_version", ""))
    if not _schema_version_compatible(got_schema, SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported dispatch.json schema_version {got_schema!r} "
            f"(expected major-compatible with {SCHEMA_VERSION!r})"
        )
    return DispatchState(
        schema_version=got_schema,
        dispatch_id=str(d["dispatch_id"]),
        started_at=str(d["started_at"]),
        ended_at=d.get("ended_at"),
        seeds=[int(s) for s in d.get("seeds") or []],
        commit_sha=str(d.get("commit_sha", "")),
        fleet=[
            FleetSnapshot(
                host=str(f["host"]),
                status=str(f.get("status", "idle")),
                current_run_id=f.get("current_run_id"),
                last_error=f.get("last_error"),
            )
            for f in (d.get("fleet") or [])
        ],
        jobs=[_job_from_dict(j) for j in (d.get("jobs") or [])],
        skipped=[_skipped_from_dict(s) for s in (d.get("skipped") or [])],
    )
```

Add `_schema_version_compatible` helper just above `_state_from_dict`:

```python
def _schema_version_compatible(got: str, expected: str) -> bool:
    """Return True iff ``got`` and ``expected`` share the same major version.

    Additive minor-version bumps (e.g. 1.0 → 1.1) must be tolerated by
    readers per Odin's schema rules in ``docs/odin/architecture.md`` §5;
    a major-version change (1.x → 2.x) is breaking and rejected.
    """
    if not got:
        return False
    try:
        return got.split(".", 1)[0] == expected.split(".", 1)[0]
    except (AttributeError, IndexError):
        return False
```

- [ ] **Step 4: Run tests to verify state.py tests pass**

Run: `python -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin`

Expected: PASS — 4 existing tests + 3 new tests = 7 total. The existing `_state` helper hardcodes `schema_version="1.0"` but that's still major-compatible, so the existing tests don't need updates.

- [ ] **Step 5: Update `tools/odin/asgard/runner.py` to thread `skipped` through**

Find every `DispatchState(...)` constructor and the `build_queue_from_env_lists` call. There are five `DispatchState(...)` sites in `runner.py` (lines ~312, ~333, ~368, plus any in tests). Each needs `skipped=...` added.

Replace the `fresh_jobs = build_queue_from_env_lists(...)` block (lines ~269-275) with:

```python
    fresh_jobs, fresh_skipped = build_queue_from_env_lists(
        physx_yaml=physx_yaml,
        newton_yaml=newton_yaml,
        seeds=options.seeds,
        dispatch_id=dispatch_id,
        include_filter=options.include_filter,
    )
```

Update the resume branch (lines ~278-293). After the existing block, preserve the resumed `skipped` from `prior_state`; if no prior state, use `fresh_skipped`:

```python
    prior_state = read_dispatch_state(dispatch_dir)
    if prior_state is not None:
        reset_in_flight_to_pending(prior_state)
        merged_jobs = _merge_jobs(prior_state.jobs, fresh_jobs)
        # Resume preserves the prior skipped[] verbatim; we don't re-evaluate.
        merged_skipped = list(prior_state.skipped)
        started_at = prior_state.started_at
        if options.retry_failed:
            retry_set = set(options.retry_failed)
            for j in merged_jobs:
                if j.run_id in retry_set and j.status == "failed":
                    j.status = "pending"
                    j.failure = None
    else:
        merged_jobs = fresh_jobs
        merged_skipped = fresh_skipped
        started_at = _utc_now_iso()
```

Add the pre-dispatch stdout block right before `_snapshot_fleet_yaml` (before line 296):

```python
    # Pre-dispatch summary of skipped (task, backend) pairs. One block per
    # (task_id, backend) combination, with all affected seeds collapsed.
    if merged_skipped:
        from collections import defaultdict

        grouped: dict[tuple[str, str], list[SkippedEntry]] = defaultdict(list)
        for sk in merged_skipped:
            grouped[(sk.task_id, sk.backend)].append(sk)
        print(f"[INFO] Skipping {len(merged_skipped)} (task, backend) pairs with no preset support:")
        for (task_id, backend), rows in sorted(grouped.items()):
            seeds_str = ", ".join(str(r.seed) for r in sorted(rows, key=lambda r: r.seed))
            avail = rows[0].presets_available
            print(f"[INFO]   {task_id} × {backend} (seeds {seeds_str}) — available: {avail}")
```

Add the `SkippedEntry` import at the top of runner.py (line ~19):

```python
from tools.odin.asgard.jobs import JobEntry, SkippedEntry, build_queue_from_env_lists
```

Update every `DispatchState(...)` constructor in `run_dispatch` to pass `skipped=merged_skipped`. Find each occurrence (there are three in `run_dispatch` — the all-down-fail bail, the partial-down-fail bail, and the seed/spawn-state). Each gets:

```python
        state = DispatchState(
            schema_version=SCHEMA_VERSION,
            dispatch_id=dispatch_id,
            ...
            jobs=merged_jobs,
            skipped=merged_skipped,
        )
```

Update the final summary print (after `state.ended_at = _utc_now_iso()` and `write_dispatch_state(...)`, ~line 437). Add a counter line right after the existing per-job summary loop. Look for the existing print pattern (or add one if missing — current `run_dispatch` doesn't print a final summary; the `verbose` flag prints per-job). Add at the very end of `run_dispatch`, before the aggregator block:

```python
    completed_n = sum(1 for j in state.jobs if j.status == "completed")
    failed_n = sum(1 for j in state.jobs if j.status == "failed")
    pending_n = sum(1 for j in state.jobs if j.status == "pending")
    skipped_n = len(state.skipped)
    skip_kinds = ", ".join(sorted({s.reason for s in state.skipped})) or "-"
    print(
        f"odin-dispatch: {completed_n} completed, {failed_n} failed, "
        f"{skipped_n} skipped ({skip_kinds}), {pending_n} pending out of "
        f"{len(state.jobs) + skipped_n} total"
    )
```

- [ ] **Step 6: Add a runner test for resume-preserves-skipped**

Append to `tools/odin/tests/test_asgard_runner.py`:

```python
from tools.odin.asgard.jobs import SkippedEntry


def test_resume_preserves_skipped_array(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    """A prior dispatch.json's skipped[] survives --resume verbatim.

    Even if the on-disk yaml has changed in the meantime, resume must
    not re-evaluate skipped — the dispatch's identity is fixed at
    first-write.
    """
    # Hand-build a dispatch.json with two skipped entries already present.
    from tools.odin.asgard.state import (
        DispatchState,
        FleetSnapshot,
        SCHEMA_VERSION,
        write_dispatch_state,
    )

    dispatch_dir = tmp_path / "20260427-120000"
    dispatch_dir.mkdir()
    seed_skipped = [
        SkippedEntry(
            task_id="Isaac-Foo-v0",
            framework="rsl_rl",
            backend="physx",
            seed=42,
            reason="preset_unsupported",
            presets_available=[],
        ),
    ]
    prior = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-120000",
        started_at="2026-04-27T12:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="",
        fleet=[FleetSnapshot(host="localhost", status="idle")],
        jobs=[],
        skipped=seed_skipped,
    )
    write_dispatch_state(dispatch_dir, prior)

    # Run a resume that produces no fresh skipped (yaml has changed since).
    physx_yaml = _write_minimal_physx_yaml(tmp_path / "physx.yaml")
    fleet = _localhost_fleet()
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True),
        ssh=stub_ssh_runner,
    )
    # The prior skipped[] is preserved exactly.
    assert len(state.skipped) == 1
    assert state.skipped[0].task_id == "Isaac-Foo-v0"
    assert state.skipped[0].reason == "preset_unsupported"
```

(If `_write_minimal_physx_yaml` or `_localhost_fleet` helpers don't already exist in `test_asgard_runner.py`, copy them from `test_asgard_integration.py`'s setup or define minimal local stubs.)

- [ ] **Step 7: Run all affected tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_asgard_state.py tools/odin/tests/test_asgard_runner.py tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin`

Expected: PASS — all of state, runner, and queue tests green.

- [ ] **Step 8: Commit**

```bash
git add tools/odin/asgard/state.py tools/odin/asgard/runner.py tools/odin/tests/test_asgard_state.py tools/odin/tests/test_asgard_runner.py
git commit -m "Asgard: dispatch.json schema 1.1 with skipped[] array

- state.py SCHEMA_VERSION 1.0 -> 1.1; validator now matches on the
  major version so 1.0 dispatch.json files keep loading on resume.
- DispatchState gains a skipped: list[SkippedEntry] field that
  round-trips atomically alongside jobs[].
- runner.py threads (jobs, skipped) through run_dispatch, preserves
  prior skipped[] on --resume verbatim, prints a pre-dispatch
  summary of skipped pairs and a final 'X completed, Y failed,
  Z skipped' counter."
```

---

## Task 7: Worker classifier — `preset_unsupported`

**Files:**
- Modify: `tools/odin/asgard/worker.py:250-281` (`_classify`)
- Test: `tools/odin/tests/test_asgard_worker.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_asgard_worker.py`:

```python
def test_worker_classifies_preset_unsupported(tmp_path: Path):
    """Stderr containing 'preset_unsupported:' maps to its own kind, not hugin_crash."""
    ssh = _FakeSSH(
        scripted={
            "hugin/run.py": SSHResult(
                exit_code=2,
                stdout="",
                stderr=(
                    "[ERROR] preset_unsupported: task 'Isaac-Foo-v0' has no "
                    "'physx' preset. Inspect raw_cfg.sim.physics.\n"
                ),
                duration_s=3.0,
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure is not None
    assert failed.failure.kind == "preset_unsupported"
    assert "missing preset" in failed.failure.message.lower()


def test_worker_falls_back_to_hugin_crash_without_marker(tmp_path: Path):
    """Regression: stderr without the marker still classifies as hugin_crash."""
    ssh = _FakeSSH(
        scripted={
            "hugin/run.py": SSHResult(
                exit_code=1, stdout="", stderr="generic crash\n", duration_s=2.0
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "hugin_crash"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_asgard_worker.py::test_worker_classifies_preset_unsupported -v --confcutdir=tools/odin`

Expected: FAIL — the worker classifier returns `hugin_crash` regardless of stderr content.

- [ ] **Step 3: Update `_classify`**

In `tools/odin/asgard/worker.py`, replace the `_classify` method (lines 250-281). The new version inserts a `preset_unsupported` branch *before* the generic `hugin_crash` fall-through:

```python
    def _classify(self, r: SSHResult, job: JobEntry, ssh_tail: Path) -> FailureInfo | None:
        if r.timed_out:
            return FailureInfo(
                kind="timeout",
                message=f"remote process exceeded {self._options.per_job_timeout_s}s",
                details={
                    "duration_s": r.duration_s,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        if r.exit_code in _INFRASTRUCTURE_DOCKER_EXIT_CODES:
            return FailureInfo(
                kind="infrastructure",
                message=(f"docker exec failed with exit {r.exit_code}: {r.stderr.strip() or 'unknown'}"),
                details={
                    "exit_code": r.exit_code,
                    "attempts": job.attempts,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        if r.exit_code != 0:
            stderr_text = r.stderr or ""
            if "preset_unsupported:" in stderr_text:
                return FailureInfo(
                    kind="preset_unsupported",
                    message="benchmark script reported missing preset",
                    details={
                        "exit_code": r.exit_code,
                        "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                    },
                )
            _last_line = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else None
            _stderr_tail = repr(_last_line) if _last_line is not None else "(empty)"
            return FailureInfo(
                kind="hugin_crash",
                message=f"exit code {r.exit_code}; stderr tail: {_stderr_tail}",
                details={
                    "exit_code": r.exit_code,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_asgard_worker.py -v --confcutdir=tools/odin`

Expected: PASS — all 13 worker tests, including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker.py
git commit -m "Worker: classify 'preset_unsupported:' stderr as its own kind

A non-zero exit with 'preset_unsupported:' anywhere in stderr is
classified as failure_kind='preset_unsupported' instead of the
catch-all 'hugin_crash'.  This is the runtime safety net that
catches yaml drift between presets_available stamping and the
upstream task's actual preset support.  All other non-zero exits
fall through to the existing hugin_crash classification."
```

---

## Task 8: Benchmark scripts fail fast on missing preset

**Files:**
- Modify: `scripts/benchmarks/benchmark_rsl_rl.py:115-122` (existing presets= injection block)
- Modify: `scripts/benchmarks/benchmark_skrl.py:111-118` (mirror)
- Test: `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py` (extend), `scripts/benchmarks/tests/test_benchmark_skrl_cli.py` (extend)

The CLI tests use a local `_inject_preset` mirror (the test files cannot import the benchmark scripts because those launch Isaac Sim). We extend both the production code and the mirror in lockstep, then test the mirror.

- [ ] **Step 1: Write failing tests for both CLI mirrors**

Append to `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`:

```python
def _inject_preset_with_validation(args_cli, hydra_args: list[str], has_physics_preset_fn) -> list[str]:
    """Mirror of the new gated injection in benchmark_rsl_rl.py.

    has_physics_preset_fn is the only injection point — the test passes
    a stub returning True / False; the production caller passes the real
    has_physics_preset(raw_cfg, name) closure.
    """
    import sys

    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    if not has_physics_preset_fn(args_cli.backend):
        sys.stderr.write(
            f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
            f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
            f"re-enumerate {{physx,newton}}_envs.yaml.\n"
        )
        sys.exit(2)
    return [f"presets={args_cli.backend}"] + hydra_args


def test_validation_blocks_unsupported_preset(capsys):
    args = _build_parser().parse_args(["--task", "Isaac-Foo-v0", "--backend", "physx"])
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        _inject_preset_with_validation(args, ["env.x=1"], has_physics_preset_fn=lambda name: False)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "preset_unsupported:" in captured.err
    assert "Isaac-Foo-v0" in captured.err


def test_validation_passes_when_supported():
    args = _build_parser().parse_args(["--task", "Isaac-Bar-v0", "--backend", "newton"])
    out = _inject_preset_with_validation(args, ["env.x=1"], has_physics_preset_fn=lambda name: True)
    assert out == ["presets=newton", "env.x=1"]


def test_validation_skipped_when_explicit_preset_present(capsys):
    """Explicit presets= in hydra_args bypasses validation (operator override)."""
    args = _build_parser().parse_args(["--task", "Isaac-Foo-v0", "--backend", "physx"])

    def _bomb(name: str) -> bool:
        raise AssertionError("validator must not run when explicit preset is present")

    out = _inject_preset_with_validation(args, ["presets=custom", "env.x=1"], has_physics_preset_fn=_bomb)
    assert out == ["presets=custom", "env.x=1"]
    assert "ignored" in capsys.readouterr().out
```

Append the same three tests to `scripts/benchmarks/tests/test_benchmark_skrl_cli.py` (with the SKRL-flavoured `_inject_preset_with_validation` mirror — identical body).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v --confcutdir=tools/odin`

Expected: FAIL — `_inject_preset_with_validation` is not defined in the test files until Step 1's edit lands; once the helper is added, the tests should pass on their own (they're testing the mirror directly). However, we still need the production change in Step 3 to stay in lockstep with the mirror.

(Adjust expectation: Step 1 adds both the helper *and* the tests, so Step 2 will actually pass. The "failing test first" pattern doesn't apply cleanly here because the mirror lives in the test file. Treat Step 2 as a sanity check that the mirror's logic is correct before Step 3 ports it to production.)

Run again to verify the mirror tests pass:

```bash
python -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v --confcutdir=tools/odin
```

Expected: PASS (all existing + new tests, including the 3 new validation tests on each side).

- [ ] **Step 3: Port the change into production benchmark scripts**

Edit `scripts/benchmarks/benchmark_rsl_rl.py`. Replace the existing block (lines 115-122):

```python
# Map --backend X to hydra presets=X so the physics preset is applied
# at config-resolve time. An explicit presets=... on the CLI wins.
if args_cli.backend is not None:
    existing_presets = [a for a in hydra_args if a.startswith("presets=")]
    if existing_presets:
        print(f"[WARNING] --backend={args_cli.backend} ignored because {existing_presets[0]} was explicitly passed.")
    else:
        hydra_args = [f"presets={args_cli.backend}"] + hydra_args
```

With:

```python
# Map --backend X to hydra presets=X so the physics preset is applied
# at config-resolve time.  Validate the request first: if the task does
# not have an X preset, exit fast with a stable stderr prefix the
# Asgard worker classifier matches on.  An explicit presets=... on
# the CLI bypasses validation (operator override).
if args_cli.backend is not None:
    existing_presets = [a for a in hydra_args if a.startswith("presets=")]
    if existing_presets:
        print(f"[WARNING] --backend={args_cli.backend} ignored because {existing_presets[0]} was explicitly passed.")
    else:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_tasks.utils.presets import has_physics_preset

        try:
            _raw_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        except Exception as exc:  # noqa: BLE001 — fall through to original behaviour
            print(
                f"[WARNING] could not load raw cfg for {args_cli.task!r} "
                f"to validate preset support ({type(exc).__name__}: {exc}); "
                f"injecting presets={args_cli.backend} unchecked.",
                file=sys.stderr,
            )
            hydra_args = [f"presets={args_cli.backend}"] + hydra_args
        else:
            if not has_physics_preset(_raw_cfg, args_cli.backend):
                sys.stderr.write(
                    f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
                    f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
                    f"re-enumerate {{physx,newton}}_envs.yaml.\n"
                )
                sys.exit(2)
            hydra_args = [f"presets={args_cli.backend}"] + hydra_args
```

Apply the identical replacement in `scripts/benchmarks/benchmark_skrl.py` (lines 111-118).

- [ ] **Step 4: Final test run**

```bash
python -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v --confcutdir=tools/odin
```

Expected: PASS — all argparse + injection-mirror tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/benchmark_rsl_rl.py scripts/benchmarks/benchmark_skrl.py \
        scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py
git commit -m "Benchmark scripts: fail fast on missing preset

When --backend X is set and the task has no X preset, the script now
writes a 'preset_unsupported:' line to stderr and exits 2.  Asgard's
worker classifier (separate commit) translates this into a
preset_unsupported failure_kind on the JobEntry — distinguishing
yaml-drift hits from real hugin crashes.  An explicit presets=X
already on the CLI still bypasses validation (operator override)."
```

---

## Task 9: Aggregator integration test for `preset_unsupported`

**Files:**
- Test: `tools/odin/tests/test_valhalla_aggregator.py` (extend)

The aggregator already passes through any `failure_kind` from the job. This task verifies that contract — no production code change, just a test that nails it down.

- [ ] **Step 1: Write a passing test that pins the contract**

Append to `tools/odin/tests/test_valhalla_aggregator.py`. The file has examples of how to construct a failing job — find the existing pattern (around line 209 with `kind="hugin_crash"`) and mirror it:

```python
def test_aggregator_passes_through_preset_unsupported_kind(tmp_path: Path):
    """A job with failure.kind=preset_unsupported flows into failures[] cleanly."""
    # Reuse the existing fixture-builder; this assumes the test file's
    # _build_minimal_dispatch_with_one_failed_job(...) helper exists.  If
    # not, copy the failed-job-only fixture from
    # test_aggregator_failed_job_yields_failures_entry (~line 209).
    dispatch_dir = tmp_path / "20260427-130000"
    dispatch_dir.mkdir()
    write_dispatch_state(
        dispatch_dir,
        DispatchState(
            schema_version="1.1",
            dispatch_id="20260427-130000",
            started_at="2026-04-27T13:00:00Z",
            ended_at="2026-04-27T13:01:00Z",
            seeds=[42],
            commit_sha="abc",
            fleet=[FleetSnapshot(host="h1", status="idle")],
            jobs=[
                JobEntry(
                    run_id="rsl-rl_physx_Isaac-Foo-v0_20260427-130000_seed42",
                    task_id="Isaac-Foo-v0",
                    framework="rsl_rl",
                    backend="physx",
                    num_envs=4096,
                    max_iterations=300,
                    seed=42,
                    bundle_dir_name="rsl-rl_physx_Isaac-Foo-v0_20260427-130000_seed42",
                    status="failed",
                    failure=FailureInfo(
                        kind="preset_unsupported",
                        message="benchmark script reported missing preset",
                        details={"exit_code": 2},
                    ),
                ),
            ],
        ),
    )
    agg = aggregate_dispatch(dispatch_dir)
    assert agg["totals"]["failed"] == 1
    f = next(f for f in agg["failures"] if f["seed"] == 42)
    assert f["failure_kind"] == "preset_unsupported"
```

If the file doesn't already import `DispatchState`, `FleetSnapshot`, `write_dispatch_state`, `JobEntry`, `FailureInfo`, add the imports. Check the top of `test_valhalla_aggregator.py` first.

- [ ] **Step 2: Run the test to verify it passes**

```bash
python -m pytest tools/odin/tests/test_valhalla_aggregator.py::test_aggregator_passes_through_preset_unsupported_kind -v --confcutdir=tools/odin
```

Expected: PASS — the aggregator's `_classify_failure` already returns `str(job_failure_kind)` verbatim; this test just pins that contract.

- [ ] **Step 3: Run the full aggregator test suite to verify no regression**

```bash
python -m pytest tools/odin/tests/test_valhalla_aggregator.py -v --confcutdir=tools/odin
```

Expected: PASS for all tests.

- [ ] **Step 4: Commit**

```bash
git add tools/odin/tests/test_valhalla_aggregator.py
git commit -m "Aggregator: pin contract that preset_unsupported flows through

The aggregator's _classify_failure passes job.failure.kind through
verbatim; no whitelist exists today and none is added.  This test
locks down that behaviour for the new preset_unsupported kind so a
future regression (e.g. someone adds a whitelist) won't silently
swallow it as 'malformed_bundle' or similar."
```

---

## Task 10: Integration test — end-to-end skipped pair

**Files:**
- Test: `tools/odin/tests/test_asgard_integration.py` (extend)

End-to-end coverage that an unsupported `(task, backend)` pair flows from yaml → queue → `dispatch.json.skipped[]` → final state without ever reaching the worker.

- [ ] **Step 1: Write the failing integration test**

Read `tools/odin/tests/test_asgard_integration.py` first to see the existing fixture shape (`stub_ssh_runner`, `stub_provisioner`, `_localhost_fleet`, etc.). Then append:

```python
def test_unsupported_pair_lands_in_skipped_array(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    """End-to-end: a (task, backend) the task doesn't support → skipped[]."""
    # Build a physx yaml with one supported and one unsupported task.
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            presets_available=["physx", "newton"],
        ),
        EnvEntry(
            task_id="Isaac-NewtonOnly-v0",
            entry_point="ep:N",
            env_cfg_entry_point="ec:N",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            presets_available=["newton"],
        ),
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    dispatch_dir = tmp_path / "20260427-140000"
    dispatch_dir.mkdir()
    fleet = _localhost_fleet()
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43], skip_aggregate=True),
        ssh=stub_ssh_runner,
    )

    # Ant ran (2 seeds); NewtonOnly skipped (2 seeds).
    assert {j.task_id for j in state.jobs} == {"Isaac-Ant-Direct-v0"}
    assert len(state.jobs) == 2
    assert {sk.task_id for sk in state.skipped} == {"Isaac-NewtonOnly-v0"}
    assert len(state.skipped) == 2
    assert all(sk.reason == "preset_unsupported" for sk in state.skipped)
    assert all(sk.backend == "physx" for sk in state.skipped)
    assert {sk.seed for sk in state.skipped} == {42, 43}

    # And dispatch.json on disk reflects the same.
    from tools.odin.asgard.state import read_dispatch_state

    reloaded = read_dispatch_state(dispatch_dir)
    assert reloaded is not None
    assert len(reloaded.skipped) == 2
    assert reloaded.skipped[0].presets_available == ["newton"]
```

Add necessary imports at the top of the file if missing: `EnvEntry`, `EnvList`, `write_env_list` from `tools.odin.common.env_list`.

- [ ] **Step 2: Run the test to verify it passes**

```bash
python -m pytest tools/odin/tests/test_asgard_integration.py::test_unsupported_pair_lands_in_skipped_array -v --confcutdir=tools/odin
```

Expected: PASS — assuming Tasks 1, 5, 6 have all landed.

- [ ] **Step 3: Run the full integration test suite**

```bash
python -m pytest tools/odin/tests/test_asgard_integration.py -v --confcutdir=tools/odin
```

Expected: PASS for all tests including the existing `test_loopback_dispatch_against_localhost`.

- [ ] **Step 4: Commit**

```bash
git add tools/odin/tests/test_asgard_integration.py
git commit -m "Integration test: unsupported (task, backend) lands in skipped[]

End-to-end coverage that a yaml row with presets_available=[newton]
+ physx dispatch produces zero JobEntry rows for that task and one
SkippedEntry per seed in dispatch.json.skipped[], without ever
spinning up a worker for it."
```

---

## Task 11: Final verification (full suite + pre-commit)

**Files:** none modified.

- [ ] **Step 1: Run the full Odin + benchmark test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ --confcutdir=tools/odin -q
```

Expected: all pass. Roughly 220+ tests total (existing + ~17 new from this plan).

- [ ] **Step 2: Pre-commit**

```bash
./isaaclab.sh -f
```

Expected: clean. If pre-commit modifies any file, review the change, stage it, and re-run.

- [ ] **Step 3: Verify commit lineage**

```bash
git log --oneline | head -12
```

Expected: 9 new commits in order (Tasks 1, 2, 3, 4 yaml regen, 5, 6, 7, 8, 9, 10) on top of the prior `5dbc1c4c797`. Task 11 has no commit of its own.

- [ ] **Step 4: Update `docs/odin/architecture.md`**

Edit `docs/odin/architecture.md`:

1. Bump the "Last updated" line (line 9):
   ```
   **Last updated:** 2026-04-27 (Odin preset-handling fix)
   ```

2. Append a change-log entry to §9 (after the most-recent entry):
   ```
   | 2026-04-27 | Odin preset-handling fix landed (spec: `docs/superpowers/specs/2026-04-27-odin-preset-handling-design.md`).  Closes the (task, backend) preset-mismatch failure mode that surfaced during T4.1 real-fleet validation: Anymal-C Flat and Quadcopter both crashed with `ValueError: Unknown preset(s): physx`.  `EnvEntry` gains `presets_available: list[str]`; the enumerator stamps it via `has_physics_preset(raw_cfg, name)`.  Asgard's `_expand_env_list` now returns `(jobs, skipped)`; unsupported pairs land in a new top-level `skipped[]` array on `dispatch.json` (schema_version 1.0 → 1.1, validator switched to major-match).  `benchmark_{rsl_rl,skrl}.py` validate the requested preset before injection and exit 2 with a `preset_unsupported:` stderr prefix; `worker._classify` maps that into `failure_kind="preset_unsupported"`.  Re-enumerated yamls now show `presets_available: []` for the two affected rows.  Backward-compatible: yaml without the new field passes through unchanged. | Odin preset-handling |
   ```

   (Pick the `Odin preset-handling` task name to match the spec/plan filename family.)

- [ ] **Step 5: Commit doc + push branch ready for review**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin preset-handling fix in architecture doc"
```

---

## Summary

**11 tasks, 10 commits**, end-to-end. Tasks 1-3 land the schema + enumerator. Task 4 regenerates yaml. Tasks 5-7 bring the dispatcher and worker online. Task 8 plants the runtime safety net at the benchmark-script layer. Tasks 9-10 nail down the contract via aggregator + integration tests. Task 11 finalises the architecture doc.

Each commit is independently reviewable; reverting any one leaves the system in a consistent state because of the "empty `presets_available` = pass through" rule and the "1.0 reads as compatible 1.1" major-match validator.

Roughly **17 new tests** plus updates to ~7 existing tests. Total expected suite size: 220+. No live-fleet validation in scope — a fresh dispatch with the regenerated yaml against the existing two-host T4.1 fleet exercises the path end-to-end and is the operator's manual verification step.
