# Odin Native-Backend Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Odin run tasks with no preset system but a known native backend (Anymal-C Flat, Quadcopter, etc.) when `--backend physx` is requested, while distinctly skipping the silent-swap case (e.g. `--backend newton` on a physx-native task) with `reason="native_backend_mismatch"` telemetry.

**Architecture:** Yaml is the source of truth — `EnvEntry.native_backend: str | None` populated by the enumerator from `type(raw_cfg.sim.physics)`. Asgard's `_expand_env_list` adds a second skip rule for the silent-swap case. Benchmark scripts skip preset injection silently when the cfg type matches the request; otherwise the existing `preset_unsupported:` exit-2 safety net catches drift.

**Tech Stack:** Python 3.10+, dataclasses, PyYAML, pytest. Touches `tools/odin/common/env_list.py`, `tools/odin/asgard/{jobs,state,runner}.py`, `scripts/benchmarks/benchmark_{rsl_rl,skrl}.py`, plus tests under `tools/odin/tests/` and `scripts/benchmarks/tests/`.

---

## File map (locked decomposition)

**Modify:**
- `tools/odin/common/env_list.py` — add `native_backend: str | None` to `EnvEntry`, extend `_ENTRY_FIELD_ORDER`, extend `_entry_from_dict` with `setdefault`, propagate field through `merge`, add `_derive_native_backend` helper near top of file, extend `build_entry_from_task_spec` with `native_backend_fn=None` injection.
- `tools/odin/asgard/jobs.py` — add `native_backend: str | None` field to `SkippedEntry`, add second skip rule in `_expand_env_list`.
- `tools/odin/asgard/state.py` — bump `SCHEMA_VERSION` `"1.1"` → `"1.2"`, extend `_skipped_to_dict` / `_skipped_from_dict` with `native_backend` round-trip.
- `tools/odin/asgard/runner.py` — extend the pre-dispatch stdout block to group by `(task_id, backend, reason)` and show `native: X` for `native_backend_mismatch`.
- `scripts/benchmarks/benchmark_rsl_rl.py` — add `_native_backend_matches` helper + new branch in the existing preset-validation block.
- `scripts/benchmarks/benchmark_skrl.py` — same change, mirrored.

**Test files to extend (no new files):**
- `tools/odin/tests/test_env_list.py`
- `tools/odin/tests/test_asgard_queue.py`
- `tools/odin/tests/test_asgard_state.py`
- `tools/odin/tests/test_asgard_runner.py`
- `tools/odin/tests/test_asgard_integration.py`
- `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`
- `scripts/benchmarks/tests/test_benchmark_skrl_cli.py`

**Data files to refresh (one commit, generated):**
- `tools/odin/config/physx_envs.yaml`
- `tools/odin/config/newton_envs.yaml`
- `tools/odin/config/newton_gap_candidates.yaml`

---

## Task 1: `EnvEntry.native_backend` field + serializer plumbing

**Files:**
- Modify: `tools/odin/common/env_list.py:150-166` (`_ENTRY_FIELD_ORDER`); `tools/odin/common/env_list.py:194-196` (`EnvEntry`); `tools/odin/common/env_list.py:218-239` (`_entry_from_dict`)
- Test: `tools/odin/tests/test_env_list.py` (extend at end of file)

- [ ] **Step 1: Write the failing tests**

Append to `tools/odin/tests/test_env_list.py`:

```python
def test_roundtrip_preserves_native_backend(tmp_path: Path):
    """native_backend list survives load + dump."""
    el = EnvList()
    el.groups["direct/quadcopter"] = [
        EnvEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/quadcopter",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            presets_available=[],
            native_backend="physx",
        )
    ]
    out = tmp_path / "envs.yaml"
    write_env_list(out, el, generator="test")
    reloaded = load_env_list(out)
    assert reloaded.groups["direct/quadcopter"][0].native_backend == "physx"


def test_load_yaml_without_native_backend_defaults_to_none(tmp_path: Path):
    """Pre-fix yaml that doesn't carry the field reads as None."""
    yaml_text = """\
schema_version: '1.0'
generator: legacy
groups:
  direct/quadcopter:
    - task_id: Isaac-Quadcopter-Direct-v0
      entry_point: ep:E
      env_cfg_entry_point: ec:E
      group: direct/quadcopter
      has_rsl_rl: true
      has_skrl: true
      has_rl_games: false
      framework: rsl_rl
      num_envs: 4096
      max_iterations: 300
      keep: true
      status: current
      suspected_gap: null
      presets_available: []
      notes: ''
"""
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_text)
    el = load_env_list(p)
    assert el.groups["direct/quadcopter"][0].native_backend is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_env_list.py::test_roundtrip_preserves_native_backend tools/odin/tests/test_env_list.py::test_load_yaml_without_native_backend_defaults_to_none -v --confcutdir=tools/odin`

Expected: FAIL — `EnvEntry.__init__() got an unexpected keyword argument 'native_backend'`.

- [ ] **Step 3: Add the field to the dataclass and serializer ordering**

Edit `tools/odin/common/env_list.py`. In `_ENTRY_FIELD_ORDER` (lines 150-166), insert `"native_backend"` between `"presets_available"` and `"notes"`. The final list:

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
    "native_backend",
    "notes",
]
```

In the `EnvEntry` dataclass (around line 194), insert `native_backend` immediately after `presets_available` and before `notes`:

```python
    presets_available: list[str] = field(default_factory=list)
    native_backend: str | None = None
    notes: str = ""
```

In `_entry_from_dict` (around line 233), add the missing-field default. Insert this line right after `known.setdefault("presets_available", [])`:

```python
    known.setdefault("native_backend", None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin` (use `./isaaclab.sh -p -m pytest ...` if isaaclab imports are needed for any pre-existing test).

Expected: PASS — including the two new tests and all existing 42 tests (44 total).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "EnvEntry: add native_backend str | None field

Stores the task's native physics backend (\"physx\", \"newton\",
\"ovphysx\", or None) for tasks with no preset system.  The Asgard
queue filter consumes this to distinguish silent-swap requests
(--backend newton on a physx-native task → skipped with reason
'native_backend_mismatch') from cleanly-runnable native pairs.
Defaults to None on read for legacy yaml — pass-through behaviour."
```

---

## Task 2: `merge` propagates `native_backend` as a derived field

**Files:**
- Modify: `tools/odin/common/env_list.py:312-402` (`merge`)
- Test: `tools/odin/tests/test_env_list.py` (extend after the existing `test_merge_*` tests)

- [ ] **Step 1: Write the failing tests**

Append to `tools/odin/tests/test_env_list.py`:

```python
def test_merge_refreshes_native_backend_on_existing_row():
    """native_backend is derived from runtime introspection — refresh it."""
    old = EnvEntry(
        task_id="Isaac-Quadcopter-Direct-v0",
        entry_point="ep:E",
        env_cfg_entry_point="ec:E",
        group="direct/quadcopter",
        has_rsl_rl=True,
        has_skrl=True,
        framework="rsl_rl",
        num_envs=4096,
        max_iterations=300,
        keep=True,
        presets_available=[],
        native_backend="newton",  # stale
    )
    new = EnvEntry(
        task_id="Isaac-Quadcopter-Direct-v0",
        entry_point="ep:E",
        env_cfg_entry_point="ec:E",
        group="direct/quadcopter",
        has_rsl_rl=True,
        has_skrl=True,
        framework="rsl_rl",
        num_envs=4096,
        max_iterations=300,
        keep=True,
        presets_available=[],
        native_backend="physx",
    )
    existing = EnvList()
    existing.groups["direct/quadcopter"] = [old]
    merged = merge(existing, [new])
    row = merged.groups["direct/quadcopter"][0]
    assert row.native_backend == "physx"


def test_merge_carries_native_backend_for_new_row():
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
        presets_available=[],
        native_backend="newton",
    )
    merged = merge(EnvList(), [new])
    row = merged.groups["direct/newtask"][0]
    assert row.status == "new"
    assert row.native_backend == "newton"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_env_list.py::test_merge_refreshes_native_backend_on_existing_row tools/odin/tests/test_env_list.py::test_merge_carries_native_backend_for_new_row -v --confcutdir=tools/odin`

Expected: FAIL — both tests assert on a field that `merge`'s `EnvEntry(...)` constructor doesn't populate, so they get the dataclass default `None`.

- [ ] **Step 3: Update `merge` to propagate the field**

In `tools/odin/common/env_list.py`, find the two `EnvEntry(...)` constructions inside `merge`:

For the "new only" branch (around line 358), add `native_backend=new.native_backend,` after `presets_available=new.presets_available,`:

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
                native_backend=new.native_backend,
            )
```

For the "old + new" branch (around line 388), add `native_backend=new.native_backend,` to the "Derived / refreshed" group (right after `presets_available=new.presets_available,`):

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
                native_backend=new.native_backend,
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

The "stale" branch (around line 404) is unchanged — it uses `**asdict(old)` which already carries the field.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin`

Expected: PASS for all `test_merge_*` tests, including the two new ones (46 total).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "env_list.merge: refresh native_backend on every pass

native_backend is derived from runtime introspection (the same way
presets_available is), so it must be refreshed on each enumeration,
not preserved from the prior yaml.  The 'old + new' branch takes
new.native_backend; the 'new only' branch carries the discovered
value verbatim; the 'stale' branch is unchanged because it uses
asdict(old) which already carries the field."
```

---

## Task 3: `_derive_native_backend` + `build_entry_from_task_spec` extension

**Files:**
- Modify: `tools/odin/common/env_list.py` — add `_derive_native_backend` near the top (after `_default_raw_cfg_loader`), extend `build_entry_from_task_spec` signature + body
- Test: `tools/odin/tests/test_env_list.py` (extend after the existing `test_build_entry_*` tests)

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_env_list.py`:

```python
def test_build_entry_native_backend_physx_when_physics_is_none():
    """SimulationCfg defaults sim.physics to None which means PhysxCfg() — treat as physx-native."""

    class _Sim:
        physics = None

    class _RawCfg:
        sim = _Sim()

    def _stub_raw_cfg_loader(task_id: str):
        return _RawCfg()

    spec = _FakeTaskSpec(
        task_id="Isaac-Quadcopter-Direct-v0",
        entry_point="isaaclab_tasks.direct.quadcopter:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "isaaclab_tasks.direct.quadcopter.cfg:Cfg",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=_stub_raw_cfg_loader,
        has_physics_preset_fn=lambda raw, name: False,
        native_backend_fn=lambda raw: "physx",
    )
    assert e.native_backend == "physx"


def test_build_entry_native_backend_newton():
    spec = _FakeTaskSpec(
        task_id="Isaac-NewtonNative-v0",
        entry_point="ep:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "ec:E",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=lambda task_id: object(),
        has_physics_preset_fn=lambda raw, name: False,
        native_backend_fn=lambda raw: "newton",
    )
    assert e.native_backend == "newton"


def test_build_entry_native_backend_none_when_preset_cfg():
    """Tasks with PresetCfg sim.physics → native_backend=None (presets_available is the source of truth)."""
    spec = _FakeTaskSpec(
        task_id="Isaac-PresetTask-v0",
        entry_point="ep:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "ec:E",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=lambda task_id: object(),
        has_physics_preset_fn=lambda raw, name: name in ("physx", "newton"),
        native_backend_fn=lambda raw: None,
    )
    assert e.native_backend is None
    assert e.presets_available == ["physx", "newton"]


def test_build_entry_native_backend_loader_failure_yields_none():
    """A raw_cfg_loader that raises leaves native_backend=None (matches presets_available behaviour)."""

    def _raises(task_id: str):
        raise RuntimeError("load failed")

    spec = _FakeTaskSpec(
        task_id="Isaac-Crashy-v0",
        entry_point="ep:E",
        kwargs={
            "rsl_rl_cfg_entry_point": "x",
            "env_cfg_entry_point": "ec:E",
        },
    )
    e = build_entry_from_task_spec(
        spec,
        defaults_loader=_noop_defaults_loader,
        raw_cfg_loader=_raises,
        has_physics_preset_fn=lambda raw, name: False,
        native_backend_fn=lambda raw: "physx",  # never called because loader raises
    )
    assert e.native_backend is None
    assert e.presets_available == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -k native_backend -v --confcutdir=tools/odin`

Expected: FAIL — `build_entry_from_task_spec` doesn't yet take `native_backend_fn` kwarg.

- [ ] **Step 3: Add `_derive_native_backend` helper + extend `build_entry_from_task_spec`**

In `tools/odin/common/env_list.py`, add the helper right after `_default_raw_cfg_loader` (near the top of the file):

```python
def _derive_native_backend(raw_cfg) -> str | None:
    """Inspect ``raw_cfg.sim.physics`` to determine the task's native backend.

    Returns:
        - ``"physx"`` if ``sim.physics`` is ``None`` (per
          :class:`~isaaclab.sim.SimulationCfg`'s docstring, ``physics=None``
          means ``PhysxCfg()``) or an instance of :class:`PhysxCfg`.
        - ``"newton"`` if ``sim.physics`` is a :class:`NewtonCfg`.
        - ``"ovphysx"`` if ``sim.physics`` is a :class:`OvPhysxCfg`.
        - ``None`` if ``sim.physics`` is a :class:`PresetCfg` subclass
          (preset system handles backend selection — ``presets_available``
          is the source of truth) or an unrecognised type.
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

Update `build_entry_from_task_spec`'s signature (find the function around line 501):

```python
def build_entry_from_task_spec(
    task_spec: Any,
    *,
    defaults_loader=load_shipped_training_defaults,
    raw_cfg_loader=_default_raw_cfg_loader,
    has_physics_preset_fn=None,
    native_backend_fn=None,
) -> EnvEntry:
```

Update the body. Find the existing block that loads `raw_cfg` and queries `presets_available`. After that block, add the native_backend derivation. Replace the `EnvEntry(...)` return at the end with one that includes `native_backend=native_backend`.

Concretely, the body section that currently has:

```python
    presets_available: list[str] = []
    try:
        raw_cfg = raw_cfg_loader(task_spec.id)
    except Exception as exc:
        print(...)
        raw_cfg = None
    if raw_cfg is not None:
        for name in ("physx", "newton"):
            try:
                if has_physics_preset_fn(raw_cfg, name):
                    presets_available.append(name)
            except Exception as exc:
                print(...)

    return EnvEntry(...)
```

Becomes:

```python
    if has_physics_preset_fn is None:
        from tools.odin.common.presets import has_physics_preset as _real_hpp

        has_physics_preset_fn = _real_hpp
    if native_backend_fn is None:
        native_backend_fn = _derive_native_backend

    presets_available: list[str] = []
    native_backend: str | None = None
    try:
        raw_cfg = raw_cfg_loader(task_spec.id)
    except Exception as exc:  # noqa: BLE001
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
        try:
            native_backend = native_backend_fn(raw_cfg)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING env_list: native_backend_fn raised for {task_spec.id}: "
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
        native_backend=native_backend,
    )
```

Also extend the function's docstring to document the new kwarg.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin`

Expected: PASS — all four new tests, plus the existing 46 (50 total). The new defaults are backward-compatible because the new kwarg has a default.

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "env_list: build_entry_from_task_spec stamps native_backend

Adds _derive_native_backend(raw_cfg) helper that inspects
type(raw_cfg.sim.physics) and returns 'physx' / 'newton' / 'ovphysx'
/ None.  Wired into build_entry_from_task_spec via a new
native_backend_fn=None injection point (defaults to
_derive_native_backend, lazy-resolved when None).  Loader / derivation
exceptions degrade to native_backend=None — same defensive shape as
the existing presets_available stamping."
```

---

## Task 4: `SkippedEntry.native_backend` + `dispatch.json` schema 1.2

**Files:**
- Modify: `tools/odin/asgard/jobs.py:42-57` (`SkippedEntry` dataclass)
- Modify: `tools/odin/asgard/state.py:34` (SCHEMA_VERSION); `tools/odin/asgard/state.py:121-141` (`_skipped_to_dict` / `_skipped_from_dict`)
- Test: `tools/odin/tests/test_asgard_state.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tools/odin/tests/test_asgard_state.py`:

```python
def test_roundtrip_skipped_with_native_backend(tmp_path: Path):
    """SkippedEntry.native_backend round-trips through dispatch.json."""
    skipped = [
        SkippedEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            framework="rsl_rl",
            backend="newton",
            seed=42,
            reason="native_backend_mismatch",
            presets_available=[],
            native_backend="physx",
        ),
    ]
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], skipped))
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded.skipped[0].native_backend == "physx"
    assert reloaded.skipped[0].reason == "native_backend_mismatch"


def test_read_skipped_without_native_backend_defaults_to_none(tmp_path: Path):
    """Skipped entries written by an older writer (no native_backend key) read with native_backend=None."""
    path = tmp_path / "dispatch.json"
    path.write_text(
        '{"schema_version": "1.1", "dispatch_id": "old", '
        '"started_at": "2026-01-01T00:00:00Z", "ended_at": null, '
        '"seeds": [42], "commit_sha": "", "fleet": [], "jobs": [], '
        '"skipped": [{"task_id": "T", "framework": "rsl_rl", "backend": "physx", '
        '"seed": 42, "reason": "preset_unsupported", "presets_available": []}]}'
    )
    s = read_dispatch_state(tmp_path)
    assert s is not None
    assert s.skipped[0].native_backend is None


def test_schema_version_writes_1_2(tmp_path: Path):
    """New dispatches write schema_version='1.2'."""
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], []))
    import json

    payload = json.loads((tmp_path / "dispatch.json").read_text())
    assert payload["schema_version"] == "1.2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_asgard_state.py -k "native_backend or schema_version_writes_1_2" -v --confcutdir=tools/odin`

Expected: FAIL — `SkippedEntry.__init__()` doesn't accept `native_backend`; `SCHEMA_VERSION` is `"1.1"`, not `"1.2"`.

- [ ] **Step 3: Update `SkippedEntry` and `state.py`**

In `tools/odin/asgard/jobs.py` (around line 42-57), extend `SkippedEntry`:

```python
@dataclass
class SkippedEntry:
    """An (task, framework, backend, seed) pair that the queue builder rejected.

    Lives next to :class:`JobEntry` because both are persisted into
    ``dispatch.json`` (jobs[] and skipped[] respectively).  ``reason``
    values today: ``"preset_unsupported"`` (yaml's
    ``presets_available`` excludes the requested backend) and
    ``"native_backend_mismatch"`` (no preset system, native backend
    doesn't match request).  Optional ``native_backend`` carries
    additional telemetry when ``reason="native_backend_mismatch"``.
    """

    task_id: str
    framework: str
    backend: str
    seed: int
    reason: str
    presets_available: list[str] = field(default_factory=list)
    native_backend: str | None = None
```

In `tools/odin/asgard/state.py`, bump version (line 34):

```python
SCHEMA_VERSION = "1.2"
```

Update `_skipped_to_dict` (around line 121) to emit the new field:

```python
def _skipped_to_dict(s: SkippedEntry) -> dict[str, Any]:
    return {
        "task_id": s.task_id,
        "framework": s.framework,
        "backend": s.backend,
        "seed": s.seed,
        "reason": s.reason,
        "presets_available": list(s.presets_available),
        "native_backend": s.native_backend,
    }
```

Update `_skipped_from_dict` (around line 132) to read the field defensively:

```python
def _skipped_from_dict(d: dict[str, Any]) -> SkippedEntry:
    return SkippedEntry(
        task_id=str(d["task_id"]),
        framework=str(d["framework"]),
        backend=str(d["backend"]),
        seed=int(d["seed"]),
        reason=str(d.get("reason", "preset_unsupported")),
        presets_available=list(d.get("presets_available") or []),
        native_backend=d.get("native_backend"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin`

Expected: PASS — 8 existing tests + 3 new tests = 11. Existing tests still pass because the major-match validator added by the preset-handling fix accepts any 1.x.

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/jobs.py tools/odin/asgard/state.py tools/odin/tests/test_asgard_state.py
git commit -m "Asgard: SkippedEntry.native_backend + dispatch.json schema 1.2

SkippedEntry gains an optional native_backend: str | None field
populated when reason='native_backend_mismatch' (telemetry: tells
the operator what the task's native backend is so they can pick
the right --backend).  state.py SCHEMA_VERSION 1.1 -> 1.2 (additive
minor bump per docs/odin/architecture.md §5; major-match validator
accepts both).  Pre-1.2 reads default the new field to None via
_skipped_from_dict's defensive .get(...)."
```

---

## Task 5: Asgard queue rule 2 — `native_backend_mismatch`

**Files:**
- Modify: `tools/odin/asgard/jobs.py:_expand_env_list` (around lines 64-130)
- Test: `tools/odin/tests/test_asgard_queue.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_asgard_queue.py`. First update `_env` helper to take an explicit `native_backend` kwarg:

```python
def _env(
    task_id: str,
    framework: str = "rsl_rl",
    keep: bool = True,
    status: str = "current",
    presets_available: list[str] | None = None,
    native_backend: str | None = None,
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
        native_backend=native_backend,
    )
```

Then append the new tests:

```python
def test_native_mismatch_skips_with_telemetry(tmp_path: Path):
    """presets_available=[] AND native_backend != requested → skipped with reason='native_backend_mismatch'."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Quadcopter-Direct-v0", presets_available=[], native_backend="physx")],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=None,
        newton_yaml=physx,
        seeds=[42, 43, 44],
        dispatch_id="20260427-100000",
    )
    assert jobs == []
    assert len(skipped) == 3
    s = skipped[0]
    assert s.task_id == "Isaac-Quadcopter-Direct-v0"
    assert s.framework == "rsl_rl"
    assert s.backend == "newton"
    assert s.reason == "native_backend_mismatch"
    assert s.presets_available == []
    assert s.native_backend == "physx"


def test_native_match_passes_through_to_runtime(tmp_path: Path):
    """presets_available=[] AND native_backend == requested → JobEntry created (runtime safety net handles injection)."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Quadcopter-Direct-v0", presets_available=[], native_backend="physx")],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 1
    assert skipped == []


def test_unknown_native_passes_through(tmp_path: Path):
    """presets_available=[] AND native_backend=None (truly unknown) → JobEntry created (runtime catches)."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Unknown-Native-v0", presets_available=[], native_backend=None)],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 1
    assert skipped == []


def test_preset_unsupported_takes_precedence_over_native(tmp_path: Path):
    """When BOTH rules could fire (presets_available populated + native mismatch), rule 1 wins.

    Theoretically a task could have a PresetCfg with no physx alternative
    but default to physx internally.  Rule 1 (preset_unsupported) is the
    correct classification — the task explicitly opts INTO the preset
    system and excludes physx from the menu.
    """
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Edge-v0", presets_available=["newton"], native_backend="physx")],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert jobs == []
    assert len(skipped) == 1
    assert skipped[0].reason == "preset_unsupported"
    # native_backend telemetry still populated for rule 1 entries.
    assert skipped[0].native_backend == "physx"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin`

Expected: FAIL on the new tests — `_expand_env_list` only has rule 1 today, so the native-mismatch case falls through to JobEntry construction.

- [ ] **Step 3: Add rule 2 to `_expand_env_list`**

In `tools/odin/asgard/jobs.py`, find `_expand_env_list` (around line 64). The function currently has rule 1 (preset gate). Replace the body's per-row block to add rule 2:

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

            # Rule 1: preset system says backend not supported.
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

            # Rule 2: no preset system, native_backend known and mismatching
            # the requested backend → silent-swap prevention.
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

Note: rule 1's SkippedEntry construction now also includes
`native_backend=row.native_backend,` (telemetry parity with rule 2).
The existing `test_unsupported_pair_skips_with_telemetry` test (from
the preset-handling spec) doesn't assert against this field, so it
keeps passing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin`

Expected: PASS — 13 prior + 4 new = 17 tests.

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/jobs.py tools/odin/tests/test_asgard_queue.py
git commit -m "Asgard: queue filter rule 2 — native_backend_mismatch

_expand_env_list adds a second skip rule that fires when
presets_available is empty AND row.native_backend is known AND
doesn't match the requested backend.  Emits SkippedEntry with
reason='native_backend_mismatch' and native_backend telemetry
populated.  Rule ordering: rule 1 (preset_unsupported) fires
first — a task with both a populated presets_available list and
a mismatching native_backend is classified by its explicit menu,
not its implicit default.  Rule 1's SkippedEntry now also carries
native_backend for telemetry parity with rule 2."
```

---

## Task 6: Runner pre-dispatch summary — show native_backend

**Files:**
- Modify: `tools/odin/asgard/runner.py` — extend the pre-dispatch stdout block in `run_dispatch`
- Test: `tools/odin/tests/test_asgard_runner.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tools/odin/tests/test_asgard_runner.py`:

```python
def test_pre_dispatch_summary_renders_native_mismatch_line(
    tmp_path: Path, stub_ssh_runner, stub_provisioner, capsys
):
    """The [INFO] block grouped by reason shows 'native: <X>' for native_backend_mismatch."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not configured")

    el = EnvList()
    el.groups["direct/quadcopter"] = [
        EnvEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/quadcopter",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=[],
            native_backend="physx",
        ),
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    dispatch_dir = tmp_path / "20260427-150000"
    dispatch_dir.mkdir()
    fleet = Fleet(
        fleet_name="loopback-test",
        hosts=[
            ValkyrieConfig(
                host="localhost",
                ssh_user=os.environ.get("USER") or "root",
                ssh_key=None,
                isaaclab_path=str(tmp_path / "remote_isaaclab"),
                container_name="loopback-container",
            ),
        ],
    )
    run_dispatch(
        fleet=fleet,
        physx_yaml=None,
        newton_yaml=physx_yaml,  # request newton on a physx-native task
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True, per_job_timeout_s=60),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )
    captured = capsys.readouterr()
    out = captured.out
    assert "native_backend_mismatch" in out
    assert "native: physx" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/odin/tests/test_asgard_runner.py::test_pre_dispatch_summary_renders_native_mismatch_line -v --confcutdir=tools/odin`

Expected: FAIL — current pre-dispatch summary doesn't surface `native:` (it groups by `(task_id, backend)` only and emits `available: ...`).

- [ ] **Step 3: Extend the pre-dispatch summary**

In `tools/odin/asgard/runner.py`, find the existing pre-dispatch summary block (the `if merged_skipped:` block, around line 300). Replace it with one that groups by `(task_id, backend, reason)` and renders the right detail per reason:

```python
    # Pre-dispatch summary of skipped (task, backend) pairs. One block per
    # (task_id, backend, reason) combination, with all affected seeds collapsed.
    if merged_skipped:
        from collections import defaultdict

        grouped: dict[tuple[str, str, str], list[SkippedEntry]] = defaultdict(list)
        for sk in merged_skipped:
            grouped[(sk.task_id, sk.backend, sk.reason)].append(sk)
        print(f"[INFO] Skipping {len(merged_skipped)} (task, backend) pairs:")
        for (task_id, backend, reason), rows in sorted(grouped.items()):
            seeds_str = ", ".join(str(r.seed) for r in sorted(rows, key=lambda r: r.seed))
            if reason == "native_backend_mismatch":
                native = rows[0].native_backend
                detail = f"native: {native}"
            else:
                avail = rows[0].presets_available
                detail = f"available: {avail}"
            print(f"[INFO]   {task_id} × {backend} (seeds {seeds_str}) — {reason} ({detail})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/odin/tests/test_asgard_runner.py -v --confcutdir=tools/odin`

Expected: PASS — 9 existing + 1 new = 10. The new test may SKIP if ssh-localhost isn't configured; that's fine.

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py tools/odin/tests/test_asgard_runner.py
git commit -m "Runner: pre-dispatch summary shows native: X for native_backend_mismatch

The [INFO] Skipping... block now groups by (task_id, backend, reason)
and renders a reason-appropriate detail: 'native: <X>' for
native_backend_mismatch, 'available: [...]' for preset_unsupported.
The final summary line (count of skipped by kind) was already
reason-aware since the preset-handling fix and needs no change."
```

---

## Task 7: Re-enumerate yaml files

**Files:**
- Modify (generated): `tools/odin/config/physx_envs.yaml`, `tools/odin/config/newton_envs.yaml`, `tools/odin/config/newton_gap_candidates.yaml`

This task contains no test code — it's a single regeneration commit that materializes `native_backend` on every existing row. Run after Tasks 1-3 have landed.

- [ ] **Step 1: Re-run the physx enumerator**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
```

Expected: Script completes; `tools/odin/config/physx_envs.yaml` is rewritten in place. The merge logic preserves human-edited fields. Every row gains a `native_backend: ...` line.

- [ ] **Step 2: Re-run the newton enumerator**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

Expected: Script completes; `tools/odin/config/newton_envs.yaml` and `tools/odin/config/newton_gap_candidates.yaml` are rewritten.

- [ ] **Step 3: Spot-check the diff**

```bash
git diff tools/odin/config/*.yaml | head -80
```

Expected: every changed row gains a `native_backend:` line. Spot-check the four shapes:

```bash
python3 -c "
import yaml
data = yaml.safe_load(open('tools/odin/config/physx_envs.yaml'))
checks = {
    'Isaac-Velocity-Flat-Anymal-C-Direct-v0': 'physx',  # no preset, physx-native
    'Isaac-Quadcopter-Direct-v0': 'physx',              # no preset, physx-native
    'Isaac-Ant-Direct-v0': None,                         # PresetCfg
    'Isaac-Velocity-Flat-G1-v0': 'newton',              # no preset, newton-native
}
results = {}
for grp in data['groups'].values():
    for r in grp:
        if r['task_id'] in checks:
            results[r['task_id']] = r['native_backend']
for tid, expected in checks.items():
    got = results.get(tid, '<missing>')
    flag = '✓' if got == expected else '✗'
    print(f'{flag} {tid}: native_backend={got!r} (expected {expected!r})')
"
```

Expected output: all four ✓. If any row prints ✗, **stop and escalate** — `_derive_native_backend` is producing the wrong answer; investigate before continuing.

- [ ] **Step 4: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/config/physx_envs.yaml tools/odin/config/newton_envs.yaml tools/odin/config/newton_gap_candidates.yaml
git commit -m "Re-enumerate env-list yamls with native_backend

One-shot regeneration after build_entry_from_task_spec learned to
stamp native_backend.  Anymal-C Flat, Quadcopter, and other no-preset
physx-native tasks land with native_backend: physx.  Velocity-Flat-G1
/ Spot / Unitree-Go1 (newton-native) land with native_backend: newton.
Tasks with PresetCfg sim.physics (Ant, Cartpole, etc.) land with
native_backend: null.  The Asgard filter rule 2 (separate commit) is
what actually consumes this."
```

---

## Task 8: Benchmark scripts — `_native_backend_matches` + skip-injection branch

**Files:**
- Modify: `scripts/benchmarks/benchmark_rsl_rl.py` (extend the existing preset-validation block)
- Modify: `scripts/benchmarks/benchmark_skrl.py` (mirror)
- Test: `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py` (extend), `scripts/benchmarks/tests/test_benchmark_skrl_cli.py` (extend)

The CLI tests use a local `_inject_preset_with_validation` mirror — extend both production and mirror in lockstep.

- [ ] **Step 1: Write failing tests for both CLI mirrors**

Append to `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`:

```python
def _inject_preset_with_validation_v2(
    args_cli,
    hydra_args: list[str],
    has_physics_preset_fn,
    native_backend_matches_fn,
) -> list[str]:
    """Mirror of the new gated injection in benchmark_rsl_rl.py (post native-backend fix).

    Two stub injection points:
      - has_physics_preset_fn(name) -> bool (existing)
      - native_backend_matches_fn(name) -> bool (new)
    """
    import sys

    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    if has_physics_preset_fn(args_cli.backend):
        return [f"presets={args_cli.backend}"] + hydra_args
    if native_backend_matches_fn(args_cli.backend):
        print(
            f"[INFO] task {args_cli.task!r} has no '{args_cli.backend}' preset; "
            f"running on native {args_cli.backend} backend (no injection).",
            file=sys.stderr,
        )
        return hydra_args
    sys.stderr.write(
        f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
        f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
        f"re-enumerate {{physx,newton}}_envs.yaml.\n"
    )
    sys.exit(2)


def test_validation_skips_injection_when_native_matches(capsys):
    """No preset, but cfg type matches request → run with no injection + [INFO] log."""
    args = _build_parser().parse_args(["--task", "Isaac-Quadcopter-Direct-v0", "--backend", "physx"])
    out = _inject_preset_with_validation_v2(
        args,
        ["env.x=1"],
        has_physics_preset_fn=lambda name: False,
        native_backend_matches_fn=lambda name: True,
    )
    assert out == ["env.x=1"]
    captured = capsys.readouterr()
    assert "running on native physx" in captured.err
    assert "no injection" in captured.err


def test_validation_still_blocks_when_native_mismatches(capsys):
    """No preset AND cfg type doesn't match → existing exit-2 + preset_unsupported: stderr (regression)."""
    args = _build_parser().parse_args(["--task", "Isaac-NewtonOnly-v0", "--backend", "physx"])
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        _inject_preset_with_validation_v2(
            args,
            ["env.x=1"],
            has_physics_preset_fn=lambda name: False,
            native_backend_matches_fn=lambda name: False,
        )
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "preset_unsupported:" in captured.err
```

Append the same two tests to `scripts/benchmarks/tests/test_benchmark_skrl_cli.py` (with the SKRL-flavoured `_inject_preset_with_validation_v2` mirror — identical body).

- [ ] **Step 2: Run tests to verify they fail / pass on the mirror**

Run: `python -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v --confcutdir=tools/odin`

Expected: PASS (the helper is local to the test file; both new tests are testing the mirror's logic). If they fail, the mirror has a bug — fix it before porting to production.

- [ ] **Step 3: Port the change into production benchmark scripts**

Edit `scripts/benchmarks/benchmark_rsl_rl.py`. Find the existing preset-validation block (around lines 115-146). Replace its `else:` branch (the part inside `if existing_presets: ... else:`) with:

```python
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
            if has_physics_preset(_raw_cfg, args_cli.backend):
                hydra_args = [f"presets={args_cli.backend}"] + hydra_args
            elif _native_backend_matches(_raw_cfg, args_cli.backend):
                print(
                    f"[INFO] task {args_cli.task!r} has no '{args_cli.backend}' "
                    f"preset; running on native {args_cli.backend} backend (no "
                    f"injection).",
                    file=sys.stderr,
                )
                # No injection — hydra_args unchanged.
            else:
                sys.stderr.write(
                    f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
                    f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
                    f"re-enumerate {{physx,newton}}_envs.yaml.\n"
                )
                sys.exit(2)
```

Add the `_native_backend_matches` helper at the top of the script's main module-level body (right after the argparse imports, before the parser definition):

```python
def _native_backend_matches(raw_cfg, requested: str) -> bool:
    """Return True iff raw_cfg.sim.physics' type matches the requested backend.

    Mirrors the introspection logic in
    :func:`tools.odin.common.env_list._derive_native_backend`.
    """
    sim = getattr(raw_cfg, "sim", None)
    if sim is None:
        return False
    physics = getattr(sim, "physics", None)
    # SimulationCfg.physics defaults to None which means PhysxCfg().
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

Apply the **identical** change in `scripts/benchmarks/benchmark_skrl.py` — same `else` replacement, same `_native_backend_matches` helper, same module-level placement.

- [ ] **Step 4: Final test run**

```bash
python -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v --confcutdir=tools/odin
```

Expected: PASS — all argparse + injection-mirror tests on both sides (10 + 14 = 24 in skrl per existing pattern, plus 2 new each = 18 in rsl, 16 in skrl after counting; the existing test counts vary by file and don't matter — just confirm nothing failed).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_rsl_rl.py scripts/benchmarks/benchmark_skrl.py \
        scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py
git commit -m "Benchmark scripts: skip injection on native cfg-type match

When --backend X is set, the task has no X preset, but the cfg's
sim.physics is the matching type (None or PhysxCfg → physx;
NewtonCfg → newton; OvPhysxCfg → ovphysx), the script now skips
preset injection silently with an [INFO] log and the task runs on
its native backend.  Mismatches still fall into the existing
preset_unsupported: exit-2 safety net.  Helper _native_backend_matches
mirrors tools/odin/common/env_list._derive_native_backend."
```

---

## Task 9: Integration test — end-to-end native_backend_mismatch

**Files:**
- Test: `tools/odin/tests/test_asgard_integration.py` (extend)

End-to-end coverage that a `native_backend_mismatch` row flows from yaml → queue → `dispatch.json.skipped[]` → on-disk reload, with the right reason and native_backend telemetry.

- [ ] **Step 1: Write the failing integration test**

Append to `tools/odin/tests/test_asgard_integration.py`:

```python
def test_native_match_runs_unsupported_pair_routes_to_skipped(
    tmp_path: Path, stub_ssh_runner, stub_provisioner
):
    """End-to-end: presets_available=[] AND native_backend != requested → skipped[]."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not configured")

    el = EnvList()
    el.groups["direct/quadcopter"] = [
        EnvEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/quadcopter",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=[],
            native_backend="physx",
        ),
    ]
    yaml_path = tmp_path / "envs.yaml"
    write_env_list(yaml_path, el, generator="test")

    dispatch_dir = tmp_path / "20260427-160000"
    dispatch_dir.mkdir()
    fleet = Fleet(
        fleet_name="loopback-test",
        hosts=[
            ValkyrieConfig(
                host="localhost",
                ssh_user=os.environ.get("USER") or "root",
                ssh_key=None,
                isaaclab_path=str(tmp_path / "remote_isaaclab"),
                container_name="loopback-container",
            ),
        ],
    )
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=None,
        newton_yaml=yaml_path,  # request newton on a physx-native task
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43], skip_aggregate=True, per_job_timeout_s=60),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Quadcopter never queued; landed in skipped[] with the new reason.
    assert state.jobs == []
    assert len(state.skipped) == 2
    assert {sk.task_id for sk in state.skipped} == {"Isaac-Quadcopter-Direct-v0"}
    assert all(sk.reason == "native_backend_mismatch" for sk in state.skipped)
    assert all(sk.backend == "newton" for sk in state.skipped)
    assert all(sk.native_backend == "physx" for sk in state.skipped)
    assert {sk.seed for sk in state.skipped} == {42, 43}

    # And dispatch.json on disk reflects the same.
    from tools.odin.asgard.state import read_dispatch_state

    reloaded = read_dispatch_state(dispatch_dir)
    assert reloaded is not None
    assert reloaded.schema_version == "1.2"
    assert len(reloaded.skipped) == 2
    assert reloaded.skipped[0].native_backend == "physx"
```

- [ ] **Step 2: Run the test to verify it passes**

```bash
python -m pytest tools/odin/tests/test_asgard_integration.py::test_native_match_runs_unsupported_pair_routes_to_skipped -v --confcutdir=tools/odin
```

Expected: PASS (or SKIP if ssh-localhost isn't configured — same skip pattern as the existing integration test).

- [ ] **Step 3: Run the full integration test suite**

```bash
python -m pytest tools/odin/tests/test_asgard_integration.py -v --confcutdir=tools/odin
```

Expected: PASS for all tests including the existing `test_loopback_dispatch_against_localhost` and `test_unsupported_pair_lands_in_skipped_array`.

- [ ] **Step 4: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_integration.py
git commit -m "Integration test: native_backend_mismatch lands in skipped[]

End-to-end coverage that a yaml row with presets_available=[] AND
native_backend='physx' + newton dispatch produces zero JobEntry rows
and one SkippedEntry per seed in dispatch.json.skipped[] with
reason='native_backend_mismatch' and native_backend='physx' carried
through.  Verifies schema_version='1.2' on the on-disk file."
```

---

## Task 10: Final verification + arch doc

**Files:** Update `docs/odin/architecture.md` only (after verification passes).

- [ ] **Step 1: Run the full Odin + benchmark test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ --confcutdir=tools/odin -q
```

Expected: all pass. Approximately 250+ tests total (235 prior + ~17 new from this plan).

If anything fails, **stop and report BLOCKED** with the failing test names.

- [ ] **Step 2: Pre-commit**

```bash
./isaaclab.sh -f
```

Expected: clean. If pre-commit modifies any file, review, stage, re-run.

- [ ] **Step 3: Verify commit lineage**

```bash
git log --oneline | head -12
```

Expected: 9 new commits in order on top of the prior `877ecf91ef8` (the preset-handling architecture-doc commit). Approximate sequence:
1. EnvEntry: add native_backend
2. env_list.merge: refresh native_backend
3. env_list: build_entry_from_task_spec stamps native_backend
4. Asgard: SkippedEntry.native_backend + dispatch.json schema 1.2
5. Asgard: queue filter rule 2 — native_backend_mismatch
6. Runner: pre-dispatch summary shows native: X
7. Re-enumerate env-list yamls with native_backend
8. Benchmark scripts: skip injection on native cfg-type match
9. Integration test: native_backend_mismatch lands in skipped[]

(plus the design-spec commit `f86561554d6` and the plan commit, both at the start of the chain.)

- [ ] **Step 4: Update `docs/odin/architecture.md`**

Edit `docs/odin/architecture.md`:

1. Bump the "Last updated" line:
   ```
   **Last updated:** 2026-04-27 (Odin native-backend routing)
   ```

2. Append a change-log entry to §9 (the markdown table near the bottom). Add this row at the end:
   ```
   | 2026-04-27 | Odin native-backend routing landed (spec: `docs/superpowers/specs/2026-04-27-odin-native-backend-design.md`).  Follow-up to the preset-handling fix: tasks with no preset system but a known native backend (Anymal-C Flat, Quadcopter, etc.) now run cleanly when the requested backend matches their native cfg type.  `EnvEntry` gains `native_backend: str | None`; the enumerator stamps it via `type(raw_cfg.sim.physics)` introspection.  Asgard's `_expand_env_list` adds a second skip rule that fires on silent-swap requests with `reason="native_backend_mismatch"` (e.g. `--backend newton` on a physx-native task).  `benchmark_{rsl_rl,skrl}.py` skip preset injection silently when the cfg type matches the request; otherwise the existing `preset_unsupported:` exit-2 safety net catches drift.  `dispatch.json` schema bumps 1.1 → 1.2 (additive; `SkippedEntry` gains optional `native_backend` field; major-match validator accepts both).  Backward-compatible: yaml without the new field reads as `native_backend=None` and falls through to the runtime safety net. | Odin native-backend routing |
   ```

- [ ] **Step 5: Commit doc**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin native-backend routing in architecture doc"
```

---

## Summary

**10 tasks, 10 commits**, end-to-end. Tasks 1-3 land the schema + enumerator. Task 4 plumbs `SkippedEntry.native_backend` + bumps `dispatch.json` schema 1.1 → 1.2. Task 5 wires the new queue rule. Task 6 tightens the runner's stdout block. Task 7 regenerates yaml. Task 8 makes the benchmark scripts respect the new "no preset, but cfg type matches" path. Task 9 adds the integration test. Task 10 finalises the architecture doc.

Each commit is independently reviewable; revertible without breaking the system because (a) `native_backend == None` falls through harmlessly until step 7 lands, (b) the major-match validator from the preset-handling fix already accepts 1.x dispatch.json files.

Roughly **17 new tests** plus updates to ~3 existing test fixtures. Total expected suite size: 250+. No live-fleet validation in scope — a fresh dispatch with the regenerated yaml against the existing two-host fleet exercises the path end-to-end and is the operator's manual verification step.
