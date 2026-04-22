# Odin T1 — Evaluation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-run benchmark pipeline for Odin: (a) a v1.0 JSON schema module + two IsaacLab benchmark scripts (RSL-RL, SKRL) + an upgraded startup profiler emitting that schema, and (b) Odin's Hugin / Munin wrapper scripts that bundle per-run artifacts. Close with a four-bundle dry-run on `Isaac-Ant-Direct-v0` covering both backends and both frameworks, committed as reference material.

**Architecture:** Three layers (per spec §3). Layer 1 (IsaacLab scripts) each emit one schema-v1 JSON to a `--output-path`. Layer 2 (Odin wrappers under `tools/odin/`) compute a `run_id`, create a bundle dir, launch one startup subprocess + one training subprocess, and write a thin `manifest.json`. Layer 3+4 (dispatcher, dashboard) are out of T1 scope.

**Tech stack:** Python 3.10+ dataclasses, argparse, subprocess, JSON. Test: `pytest`. Upstream scripts live under `scripts/benchmarks/`. Odin code lives at `tools/odin/` (moves out when Odin graduates). Schema module lives at `source/isaaclab/isaaclab/test/benchmark/standard_schema.py`.

**Spec reference:** `docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md`

**Architecture reference (living):** `docs/odin/architecture.md`

**Branch:** `antoiner/feat/odin` (local commits only; do not push).

---

## Preamble — conventions used in every task

- All commits use the IsaacLab commit-message style from `AGENTS.md`: imperative mood, capitalized subject ≤ ~50 chars, blank line before body, no AI co-author lines.
- `./isaaclab.sh -f` must pass before every commit. Not after.
- Tests run sequentially, never in parallel (per memory: GPU tests in parallel segfault).
- New Python files get the SPDX 2022-2026 header:
  ```python
  # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```
- Markdown files under `docs/odin/` and `tools/odin/` do not get SPDX headers (existing `docs/` files do not carry them either).
- Each major task ends with a commit; do not batch unrelated tasks into one commit.

---

## Phase 1 — Schema foundation

### Task 1: Create `standard_schema.py` — dataclasses + writer

This is the canonical schema definition. Every downstream consumer reads it; every benchmark writer populates it. Keep it dependency-free (stdlib only) so it loads fast and can be imported in tests without dragging IsaacLab.

**Files:**
- Create: `source/isaaclab/isaaclab/test/benchmark/standard_schema.py`

- [ ] **Step 1: Write the module skeleton with dataclasses**

Create `source/isaaclab/isaaclab/test/benchmark/standard_schema.py` with the v1.0 schema as frozen dataclasses, plus an enum for `status` and `RunKind`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin standard benchmark schema (v1.0).

Defines the on-disk JSON schema shared by IsaacLab's training and startup
benchmark scripts. Producers populate one of the ``*Bundle`` dataclasses and
call :func:`write_bundle_file` to emit schema-compliant JSON. Consumers
(dashboards, validators) read the same file and reconstruct the dataclasses.

The schema is designed so each file is self-contained: every ``*Bundle``
carries its own ``versions`` and ``hardware`` metadata so a reader need not
cross-reference other files in the bundle directory.

Current version: 1.0
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "1.0"

Framework = Literal["rsl_rl", "skrl"]
Backend = Literal["physx", "newton"]
RunStatus = Literal["completed", "interrupted", "crashed"]


@dataclass(frozen=True)
class MeanStd:
    """Scalar with mean and standard deviation."""
    mean: float
    std: float


@dataclass(frozen=True)
class MeanStdPeak:
    """Scalar with mean, standard deviation, and peak."""
    mean: float
    std: float
    peak: float


@dataclass(frozen=True)
class GpuDeviceInfo:
    name: str
    mem_gb: float
    compute_cap: str


@dataclass(frozen=True)
class Hardware:
    hostname: str
    gpu_devices: list[GpuDeviceInfo]
    cpu_name: str
    cpu_count: int
    ram_gb: float


@dataclass(frozen=True)
class Versions:
    """Software versions captured at run time. Framework-specific fields (rsl_rl,
    skrl) are null when not used by the run."""
    isaaclab: str
    isaacsim: str | None
    kit: str | None
    newton: str | None
    warp: str | None
    mjwarp: str | None
    torch: str
    rsl_rl: str | None
    skrl: str | None
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    framework: Framework
    backend: Backend
    task: str
    seed: int
    num_envs: int
    max_iterations: int
    start_time_utc: str       # ISO-8601 Z-suffixed
    end_time_utc: str
    duration_s: float
    status: RunStatus


@dataclass(frozen=True)
class StartupPhaseTimes:
    app_launch: float
    env_creation: float
    first_step: float
    python_imports: float | None = None
    task_config: float | None = None


@dataclass(frozen=True)
class Runtime:
    startup_phase_times_s: StartupPhaseTimes
    iterations_completed: int
    total_wall_time_s: float
    steps_per_iteration: int
    iteration_time_s: MeanStd
    env_steps_per_s: MeanStd
    iterations_per_s: MeanStd


@dataclass(frozen=True)
class Resources:
    gpu_util_pct: MeanStd
    gpu_mem_gb: MeanStdPeak
    cpu_util_pct: MeanStd
    ram_gb: MeanStdPeak


@dataclass(frozen=True)
class LearningCurve:
    final_raw: float
    final_ema: float
    series_per_iter: list[float] | None  # None when --no-series


@dataclass(frozen=True)
class Learning:
    ema_alpha: float
    reward: LearningCurve
    ep_length: LearningCurve


@dataclass(frozen=True)
class TrainingBundle:
    """Top-level shape of training.json."""
    run: RunIdentity
    versions: Versions
    hardware: Hardware
    runtime: Runtime
    resources: Resources
    learning: Learning
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CProfileFunction:
    name: str
    own_time_s: float
    cum_time_s: float
    calls: int


@dataclass(frozen=True)
class StartupPhase:
    total_time_s: float
    top_functions: list[CProfileFunction]


@dataclass(frozen=True)
class StartupConfig:
    top_n: int
    whitelist: str | None


@dataclass(frozen=True)
class StartupRunIdentity:
    """Startup runs omit num_envs/max_iterations (not meaningful)."""
    run_id: str
    framework: Framework
    backend: Backend
    task: str
    seed: int
    start_time_utc: str
    end_time_utc: str
    duration_s: float
    status: RunStatus


@dataclass(frozen=True)
class StartupBundle:
    """Top-level shape of startup.json."""
    run: StartupRunIdentity
    versions: Versions
    hardware: Hardware
    phases: dict[str, StartupPhase]
    config: StartupConfig
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class ManifestPhase:
    file: str
    status: RunStatus | Literal["failed"]
    duration_s: float
    exit_code: int


@dataclass(frozen=True)
class ManifestConfig:
    framework: Framework
    backend: Backend
    task: str
    seed: int
    num_envs: int
    max_iterations: int


@dataclass(frozen=True)
class ManifestMachine:
    hostname: str
    git_commit: str | None
    git_branch: str | None


@dataclass(frozen=True)
class Manifest:
    """Top-level shape of manifest.json (Odin-side)."""
    run_id: str
    run_start_time_utc: str
    run_end_time_utc: str
    run_duration_s: float
    config: ManifestConfig
    machine: ManifestMachine
    phases: dict[str, ManifestPhase]
    artifacts: list[str]
    schema_version: str = SCHEMA_VERSION


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def write_bundle_file(bundle: TrainingBundle | StartupBundle | Manifest, path: str) -> None:
    """Write a bundle dataclass to disk as schema-v1 JSON.

    Creates the parent directory if missing. Uses ``indent=2`` for readability;
    payloads are small (~10 KB training.json, ~50 KB startup.json).

    Args:
        bundle: One of :class:`TrainingBundle`, :class:`StartupBundle`, or :class:`Manifest`.
        path: Absolute or relative path to the output file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_plain(bundle), f, indent=2, sort_keys=False)
        f.write("\n")
```

- [ ] **Step 2: Run existing tests to make sure nothing else broke**

Run: `./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/ -x -q`
Expected: all existing tests still pass (no new tests yet, but we must not break the existing benchmark tests).

- [ ] **Step 3: Run formatter/linter**

Run: `./isaaclab.sh -f`
Expected: no modifications to the new file (or only auto-fix trailing whitespace). If pre-commit modifies the file, review and re-run.

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab/isaaclab/test/benchmark/standard_schema.py
git commit -m "Add Odin v1.0 benchmark standard schema module"
```

---

### Task 2: Unit tests for `standard_schema`

**Files:**
- Create: `source/isaaclab/test/benchmark/test_standard_schema.py`

- [ ] **Step 1: Write the failing test for round-trip of a minimal TrainingBundle**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Odin v1.0 benchmark standard schema."""

import json
import os

import pytest

from isaaclab.test.benchmark.standard_schema import (
    SCHEMA_VERSION,
    GpuDeviceInfo,
    Hardware,
    Learning,
    LearningCurve,
    MeanStd,
    MeanStdPeak,
    Resources,
    RunIdentity,
    Runtime,
    StartupPhaseTimes,
    TrainingBundle,
    Versions,
    write_bundle_file,
)


def _minimal_training_bundle() -> TrainingBundle:
    """Construct a valid TrainingBundle with placeholder numeric values."""
    return TrainingBundle(
        run=RunIdentity(
            run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
            framework="rsl_rl",
            backend="physx",
            task="Isaac-Ant-Direct-v0",
            seed=42,
            num_envs=4096,
            max_iterations=500,
            start_time_utc="2026-04-22T13:15:00Z",
            end_time_utc="2026-04-22T13:47:22Z",
            duration_s=1942.1,
            status="completed",
        ),
        versions=Versions(
            isaaclab="4.6.8", isaacsim="5.0.0", kit="107.1.0",
            newton="0.1.2", warp="1.7.3", mjwarp="0.0.4",
            torch="2.5.1", rsl_rl="2.3.0", skrl=None,
            git_commit="3d42b11d513", git_branch="antoiner/feat/odin", git_dirty=False,
        ),
        hardware=Hardware(
            hostname="valkyrie-03",
            gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
            cpu_name="AMD EPYC 7763", cpu_count=64, ram_gb=512.0,
        ),
        runtime=Runtime(
            startup_phase_times_s=StartupPhaseTimes(app_launch=18.4, env_creation=22.9, first_step=4.1),
            iterations_completed=500, total_wall_time_s=1946.0, steps_per_iteration=24,
            iteration_time_s=MeanStd(mean=3.82, std=0.04),
            env_steps_per_s=MeanStd(mean=1_071_780.0, std=11_200.0),
            iterations_per_s=MeanStd(mean=0.2618, std=0.0028),
        ),
        resources=Resources(
            gpu_util_pct=MeanStd(mean=87.2, std=6.1),
            gpu_mem_gb=MeanStdPeak(mean=18.4, std=0.3, peak=19.2),
            cpu_util_pct=MeanStd(mean=31.5, std=4.8),
            ram_gb=MeanStdPeak(mean=22.1, std=0.4, peak=24.8),
        ),
        learning=Learning(
            ema_alpha=0.05,
            reward=LearningCurve(final_raw=1823.4, final_ema=1796.1, series_per_iter=[12.3, 34.5, 58.1]),
            ep_length=LearningCurve(final_raw=987.0, final_ema=962.3, series_per_iter=[4.1, 5.0, 7.2]),
        ),
    )


def test_training_bundle_round_trip(tmp_path):
    """Writing a TrainingBundle and reloading via json gives back identical data."""
    bundle = _minimal_training_bundle()
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle, path)

    with open(path) as f:
        data = json.load(f)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["run"]["run_id"] == bundle.run.run_id
    assert data["runtime"]["env_steps_per_s"]["mean"] == pytest.approx(1_071_780.0)
    assert data["resources"]["ram_gb"]["peak"] == pytest.approx(24.8)
    assert data["learning"]["reward"]["series_per_iter"] == [12.3, 34.5, 58.1]
    assert data["versions"]["skrl"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_standard_schema.py::test_training_bundle_round_trip -v`
Expected: PASS (the module already exists from Task 1). If it FAILS, diagnose before moving on.

- [ ] **Step 3: Add the "no series" case**

Append to the test file:

```python
def test_training_bundle_without_series(tmp_path):
    """With series_per_iter=None, the JSON contains an explicit null."""
    bundle = _minimal_training_bundle()
    bundle_no_series = dataclasses.replace(
        bundle,
        learning=Learning(
            ema_alpha=0.05,
            reward=LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None),
            ep_length=LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None),
        ),
    )
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle_no_series, path)
    with open(path) as f:
        data = json.load(f)
    assert data["learning"]["reward"]["series_per_iter"] is None
    assert data["learning"]["ep_length"]["series_per_iter"] is None
```

Also add `import dataclasses` at the top of the file.

- [ ] **Step 4: Add a StartupBundle round-trip test**

```python
from isaaclab.test.benchmark.standard_schema import (
    CProfileFunction,
    StartupBundle,
    StartupConfig,
    StartupPhase,
    StartupRunIdentity,
)


def test_startup_bundle_round_trip(tmp_path):
    """StartupBundle round-trips with phase dict and top-function lists."""
    bundle = StartupBundle(
        run=StartupRunIdentity(
            run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
            framework="rsl_rl", backend="physx", task="Isaac-Ant-Direct-v0",
            seed=42, start_time_utc="2026-04-22T13:15:00Z",
            end_time_utc="2026-04-22T13:15:48Z", duration_s=48.7, status="completed",
        ),
        versions=_minimal_training_bundle().versions,
        hardware=_minimal_training_bundle().hardware,
        phases={
            "app_launch": StartupPhase(total_time_s=18.4, top_functions=[
                CProfileFunction(name="isaaclab.x:y", own_time_s=1.8, cum_time_s=2.4, calls=4312)
            ]),
            "env_creation": StartupPhase(total_time_s=22.9, top_functions=[]),
            "first_step": StartupPhase(total_time_s=4.1, top_functions=[]),
        },
        config=StartupConfig(top_n=30, whitelist="startup_whitelist.yaml"),
    )
    path = os.path.join(tmp_path, "startup.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["phases"]["app_launch"]["total_time_s"] == pytest.approx(18.4)
    assert data["phases"]["app_launch"]["top_functions"][0]["calls"] == 4312
```

- [ ] **Step 5: Add a Manifest round-trip test**

```python
from isaaclab.test.benchmark.standard_schema import (
    Manifest,
    ManifestConfig,
    ManifestMachine,
    ManifestPhase,
)


def test_manifest_round_trip(tmp_path):
    m = Manifest(
        run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
        run_start_time_utc="2026-04-22T13:15:00Z",
        run_end_time_utc="2026-04-22T13:47:48Z",
        run_duration_s=1968.3,
        config=ManifestConfig(framework="rsl_rl", backend="physx",
                              task="Isaac-Ant-Direct-v0", seed=42,
                              num_envs=4096, max_iterations=500),
        machine=ManifestMachine(hostname="valkyrie-03", git_commit="abc",
                                git_branch="antoiner/feat/odin"),
        phases={
            "startup":  ManifestPhase(file="startup.json",  status="completed",
                                      duration_s=48.7, exit_code=0),
            "training": ManifestPhase(file="training.json", status="completed",
                                      duration_s=1942.1, exit_code=0),
        },
        artifacts=["manifest.json", "startup.json", "training.json"],
    )
    path = os.path.join(tmp_path, "manifest.json")
    write_bundle_file(m, path)
    with open(path) as f:
        data = json.load(f)
    assert data["phases"]["training"]["exit_code"] == 0
    assert data["schema_version"] == SCHEMA_VERSION
```

- [ ] **Step 6: Run all tests together**

Run: `./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_standard_schema.py -v`
Expected: 4 tests PASS.

- [ ] **Step 7: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/test/benchmark/test_standard_schema.py
git commit -m "Add unit tests for Odin standard schema"
```

---

## Phase 2 — IsaacLab benchmark script upgrades

### Task 3: Extend `benchmark_startup.py` to emit v1.0 `startup.json`

The existing script already captures 5 phases (`app_launch`, `python_imports`, `task_config`, `env_creation`, `first_step`) and their top cProfile functions. We add an `--output-path` that, when set, writes a `StartupBundle` in addition to / instead of the legacy backend output.

**Files:**
- Modify: `scripts/benchmarks/benchmark_startup.py`

- [ ] **Step 1: Read the current script end-to-end**

Read `scripts/benchmarks/benchmark_startup.py` top to bottom. Identify:
1. The place where `phases` dict is built (currently inside `main()`).
2. Where `benchmark._finalize_impl()` is called.
3. How `args_cli.output_path` is currently used.

You're adding: a new CLI flag to toggle v1 emission, a helper to build the `StartupBundle` from the existing phase data, and a call to `write_bundle_file()` after the existing `_finalize_impl()`.

- [ ] **Step 2: Add the `--schema_v1_output` CLI argument**

In the argparse block (around existing `--output_path`), add:

```python
parser.add_argument(
    "--schema_v1_output",
    type=str,
    default=None,
    help="If set, write a schema-v1 startup.json to this path (Odin bundle format).",
)
```

Leaves the existing `--output_path` + `--benchmark_backend` untouched, so standalone use is unaffected.

- [ ] **Step 3: Add a small builder function**

Near the bottom of the script (above `if __name__ == "__main__":`), add:

```python
def _build_startup_bundle(
    phases_data: dict,
    run_start: datetime,
    run_end: datetime,
    status: "RunStatus",
) -> "StartupBundle":
    """Build a schema-v1 StartupBundle from the collected phase data.

    Args:
        phases_data: The same ``phases`` dict ``main()`` built for legacy logging.
        run_start: UTC timestamp when the whole script started.
        run_end: UTC timestamp when the whole script finished.
        status: Completion status of the run.

    Returns:
        A :class:`StartupBundle` ready to be passed to :func:`write_bundle_file`.
    """
    from isaaclab.test.benchmark.standard_schema import (
        CProfileFunction,
        StartupBundle,
        StartupConfig,
        StartupPhase,
        StartupRunIdentity,
    )

    # Infer framework: None at startup level — pick 'rsl_rl' as a schema placeholder
    # (the Odin wrapper provides the real framework via the run_id it passes in).
    # TODO(T3 execution): resolve framework from the run_id if it was passed.
    framework = "rsl_rl"

    # Backend: read from env_cfg.sim.backend if available. For MVP, assume 'physx'
    # unless the env was built with a Newton sim context.
    backend = _infer_backend_from_args_cli(args_cli)  # helper added below

    phases = {}
    for name, data in phases_data.items():
        top_funcs = []
        for label, tottime_ms, cumtime_ms in parse_cprofile_stats(
            data["profile"], _ISAACLAB_PREFIXES, top_n=args_cli.top_n,
            whitelist=_WHITELIST.get(name),
        ):
            top_funcs.append(CProfileFunction(
                name=label,
                own_time_s=tottime_ms / 1000.0,
                cum_time_s=cumtime_ms / 1000.0,
                calls=0,  # parse_cprofile_stats does not currently return calls; leave 0 for v1
            ))
        phases[name] = StartupPhase(total_time_s=data["wall_clock_ms"] / 1000.0,
                                    top_functions=top_funcs)

    run_id = args_cli.run_id if getattr(args_cli, "run_id", None) else _synth_run_id(framework, backend)

    return StartupBundle(
        run=StartupRunIdentity(
            run_id=run_id, framework=framework, backend=backend,
            task=args_cli.task, seed=args_cli.seed or 0,
            start_time_utc=run_start.isoformat().replace("+00:00", "Z"),
            end_time_utc=run_end.isoformat().replace("+00:00", "Z"),
            duration_s=(run_end - run_start).total_seconds(),
            status=status,
        ),
        versions=_capture_versions(benchmark),
        hardware=_capture_hardware(benchmark),
        phases=phases,
        config=StartupConfig(top_n=args_cli.top_n, whitelist=args_cli.whitelist_config),
    )
```

Define the helpers `_infer_backend_from_args_cli`, `_synth_run_id`, `_capture_versions`, `_capture_hardware` near the top of the file (below imports). `_capture_versions` / `_capture_hardware` extract the `Versions` / `Hardware` dataclasses from the benchmark's already-populated `_manual_recorders`. Reference implementations:

```python
def _capture_versions(bm) -> "Versions":
    from isaaclab.test.benchmark.standard_schema import Versions
    vi = bm._manual_recorders["VersionInfo"].get_data()
    meta = {m.name: m.data for m in vi.metadata}
    return Versions(
        isaaclab=meta.get("isaaclab_version", "unknown"),
        isaacsim=meta.get("isaacsim_version"),
        kit=meta.get("kit_version"),
        newton=meta.get("newton_version"),
        warp=meta.get("warp_version"),
        mjwarp=meta.get("mujoco_warp_version") or meta.get("mjwarp_version"),
        torch=meta.get("torch_version", "unknown"),
        rsl_rl=meta.get("rsl_rl_version"),
        skrl=meta.get("skrl_version"),
        git_commit=meta.get("git_commit"),
        git_branch=meta.get("git_branch"),
        git_dirty=bool(meta.get("git_dirty", False)),
    )


def _capture_hardware(bm) -> "Hardware":
    from isaaclab.test.benchmark.standard_schema import GpuDeviceInfo, Hardware
    gpu = bm._manual_recorders["GPUInfo"].get_data()
    cpu = bm._manual_recorders["CPUInfo"].get_data()
    mem = bm._manual_recorders["MemoryInfo"].get_data()
    gpu_meta = {m.name: m.data for m in gpu.metadata}
    cpu_meta = {m.name: m.data for m in cpu.metadata}
    mem_meta = {m.name: m.data for m in mem.metadata}
    devices_raw = gpu_meta.get("gpu_devices", [])
    devices = [GpuDeviceInfo(
        name=d.get("name", "unknown"),
        mem_gb=float(d.get("mem_gb", 0.0)),
        compute_cap=str(d.get("compute_cap", "unknown")),
    ) for d in devices_raw]
    import socket
    return Hardware(
        hostname=socket.gethostname(),
        gpu_devices=devices,
        cpu_name=str(cpu_meta.get("cpu_name", "unknown")),
        cpu_count=int(cpu_meta.get("cpu_count", 0) or 0),
        ram_gb=float(mem_meta.get("ram_gb", 0.0) or 0.0),
    )


def _synth_run_id(framework: str, backend: str) -> str:
    """Fallback run_id when --run_id not provided (standalone use)."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fw = framework.replace("_", "-")
    seed = args_cli.seed or 0
    return f"{fw}_{backend}_{args_cli.task}_{stamp}_seed{seed}"


def _infer_backend_from_args_cli(args) -> str:
    """Infer backend from --device/simulation args. Defaults to 'physx'."""
    # For v1 we read an explicit --backend if present, else fall back to 'physx'.
    return getattr(args, "backend", None) or "physx"
```

Also add a `--backend {physx,newton}` and `--run_id` CLI arg so Odin can pass them explicitly:

```python
parser.add_argument("--backend", choices=["physx", "newton"], default=None,
                    help="Physics backend tag recorded in the bundle. Odin wrappers pass this; "
                         "if omitted, defaults to 'physx'.")
parser.add_argument("--run_id", type=str, default=None,
                    help="Run identity string to embed in the bundle. Odin wrappers pass this.")
```

Note on the `calls` field being `0`: `parse_cprofile_stats` in `scripts/benchmarks/utils.py` does not currently return call counts. For v1 we accept the placeholder; a follow-up can extend `parse_cprofile_stats` to return calls. Record this as a known limitation in `docs/odin/architecture.md` §9 when touching that file later.

- [ ] **Step 4: Wire the emission into `main()`**

Inside `main()`, immediately after `benchmark._finalize_impl()` succeeds, add:

```python
# v1 schema emission (Odin path)
if args_cli.schema_v1_output is not None:
    from datetime import datetime, timezone
    from isaaclab.test.benchmark.standard_schema import write_bundle_file
    run_end = datetime.now(timezone.utc)
    # run_start is the app-launch begin timestamp captured at top of script
    run_start_dt = datetime.fromtimestamp(_APP_START_WALL / 1e9, tz=timezone.utc)
    bundle = _build_startup_bundle(phases, run_start_dt, run_end, status="completed")
    write_bundle_file(bundle, args_cli.schema_v1_output)
```

At module top (after the `app_launch_time_begin = ...` line near the bottom of the file), capture `_APP_START_WALL`:

```python
import time as _time
_APP_START_WALL = _time.time_ns()
```

If the script crashes partway (e.g. env.reset raises), wrap the emission in a `try/finally` that still writes a `status="crashed"` bundle with whatever phases completed. Pattern:

```python
status = "completed"
try:
    # ... existing main body ...
except Exception:
    status = "crashed"
    raise
finally:
    if args_cli.schema_v1_output is not None:
        # build bundle from whatever phases data we have; use empty dict if none yet
        bundle = _build_startup_bundle(
            phases if "phases" in dir() else {},
            run_start_dt, datetime.now(timezone.utc), status=status,
        )
        write_bundle_file(bundle, args_cli.schema_v1_output)
```

- [ ] **Step 5: Smoke-test the script still runs without `--schema_v1_output`**

Run (from repo root):
```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
    --task Isaac-Ant-Direct-v0 --num_envs 64 --headless --top_n 5
```
Expected: runs to completion, emits the normal backend JSON, no crashes. (The Ant Direct task must be registered; if it errors on import, stop and investigate before continuing.)

- [ ] **Step 6: Smoke-test v1 emission with `--schema_v1_output`**

Run:
```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
    --task Isaac-Ant-Direct-v0 --num_envs 64 --headless --top_n 5 \
    --backend physx --run_id test-startup-smoke \
    --schema_v1_output /tmp/odin_smoke/startup.json
```
Expected: `/tmp/odin_smoke/startup.json` exists, parses as JSON, top-level keys include `schema_version`, `run`, `versions`, `hardware`, `phases`, `config`. Confirm with:

```bash
./isaaclab.sh -p -c "import json; d = json.load(open('/tmp/odin_smoke/startup.json')); \
    print(d['schema_version'], d['run']['run_id'], list(d['phases']))"
```

Expected output: `1.0 test-startup-smoke ['app_launch', 'python_imports', 'task_config', 'env_creation', 'first_step']`

- [ ] **Step 7: Format and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_startup.py
git commit -m "Emit v1.0 startup bundle from benchmark_startup.py"
```

---

### Task 4: Extend `benchmark_rsl_rl.py` to emit v1.0 `training.json`

The existing script already computes reward / ep_length series post-hoc from tensorboard via `parse_tf_logs`, captures startup timings, and has recorders attached. We add: EMA computation, the run-identity / schema-v1 bundle builder, a `--schema_v1_output` CLI flag, and matching `--backend` / `--run_id` / `--ema_alpha` / `--no_series`.

**Files:**
- Modify: `scripts/benchmarks/benchmark_rsl_rl.py`

- [ ] **Step 1: Re-read the script top to bottom**

Start with `main()` (from line 139 onwards). Identify:
1. The existing `parse_tf_logs(log_dir)` call that produces `log_data["Train/mean_reward"]` and `log_data["Train/mean_episode_length"]` — these are the per-iteration series.
2. The existing `log_runtime_step_times` call and its `rl_training_times` dict — this already carries collection/learning times and a `Total FPS` list.
3. The existing `benchmark._finalize_impl()` call.

This is where you'll attach the v1 emission.

- [ ] **Step 2: Add a small EMA helper above `main()`**

```python
def _compute_ema(series: list[float], alpha: float) -> float:
    """Exponentially weighted moving average over a per-iteration series.

    Returns the final EMA value: ``x_0`` initialised to ``series[0]`` and updated
    as ``x_t = alpha * y_t + (1 - alpha) * x_{t-1}``. Empty series returns 0.0.

    Args:
        series: Per-iteration scalar values (reward or episode length).
        alpha: Smoothing factor in [0, 1]. Smaller values give more smoothing.

    Returns:
        Final EMA value after walking the full series.
    """
    if not series:
        return 0.0
    ema = float(series[0])
    for y in series[1:]:
        ema = alpha * float(y) + (1.0 - alpha) * ema
    return ema
```

Add a matching unit test in `source/isaaclab/test/benchmark/test_standard_schema.py` (or a new `test_ema.py` next to it) — TDD for the numerical behaviour:

```python
def test_compute_ema_basic():
    # EMA with alpha=0.5 on [1, 3, 5]: 1 -> 2 -> 3.5
    from scripts.benchmarks.benchmark_rsl_rl import _compute_ema
    assert _compute_ema([1, 3, 5], alpha=0.5) == pytest.approx(3.5)
    assert _compute_ema([], alpha=0.1) == 0.0
    assert _compute_ema([42.0], alpha=0.1) == pytest.approx(42.0)
```

(Requires `scripts/` to be importable; if the pytest import path doesn't see `scripts/`, skip this unit test and cover EMA via a test in `tools/odin/tests/` where we have more flexibility.)

- [ ] **Step 3: Add the four new CLI flags**

Below the existing `--output_path`:

```python
parser.add_argument("--backend", choices=["physx", "newton"], default=None,
                    help="Physics backend tag recorded in the bundle.")
parser.add_argument("--run_id", type=str, default=None,
                    help="Run identity string. Odin wrappers pass this; "
                         "if omitted, a synthetic run_id is generated.")
parser.add_argument("--schema_v1_output", type=str, default=None,
                    help="If set, write schema-v1 training.json to this path.")
parser.add_argument("--ema_alpha", type=float, default=0.05,
                    help="EMA smoothing factor for reward/ep_length (default 0.05, ~20-sample window).")
parser.add_argument("--no_series", action="store_true", default=False,
                    help="Omit per-iteration series from training.json (leaves final_raw + final_ema only).")
```

- [ ] **Step 4: Add `_build_training_bundle()` helper near the top**

Symmetric to `_build_startup_bundle` from Task 3. Pulls from existing data structures:

```python
def _build_training_bundle(
    log_data, rl_training_times, agent_cfg, env, args,
    benchmark, run_start_dt, run_end_dt, status: str,
    app_launch_ms: float, env_creation_ms: float, first_step_ms: float,
):
    from isaaclab.test.benchmark.standard_schema import (
        Hardware, Learning, LearningCurve, MeanStd, MeanStdPeak, Resources,
        RunIdentity, Runtime, StartupPhaseTimes, TrainingBundle, Versions,
    )
    import numpy as np

    reward_series = [float(x) for x in log_data.get("Train/mean_reward", [])]
    ep_len_series = [float(x) for x in log_data.get("Train/mean_episode_length", [])]

    # Iteration time from total_fps: iter_time = num_envs * steps_per_iter / total_fps
    total_fps = log_data.get("Perf/total_fps", [])
    num_envs = env.unwrapped.num_envs
    steps_per_iter = agent_cfg.num_steps_per_env
    iter_times = [num_envs * steps_per_iter / fps if fps > 0 else 0.0 for fps in total_fps]

    def _ms(xs):
        return MeanStd(mean=float(np.mean(xs)) if xs else 0.0,
                       std=float(np.std(xs)) if xs else 0.0)

    env_steps_per_s_series = [num_envs * steps_per_iter / t if t > 0 else 0.0 for t in iter_times]
    iters_per_s_series = [1.0 / t if t > 0 else 0.0 for t in iter_times]

    # Resource aggregates: pull from BenchmarkMonitor / recorders via benchmark._manual_recorders
    resources = _capture_resources(benchmark)

    run_id = args.run_id or _synth_run_id("rsl_rl", args.backend or "physx", args.task, args.seed)

    return TrainingBundle(
        run=RunIdentity(
            run_id=run_id,
            framework="rsl_rl",
            backend=args.backend or "physx",
            task=args.task,
            seed=args.seed,
            num_envs=num_envs,
            max_iterations=agent_cfg.max_iterations,
            start_time_utc=run_start_dt.isoformat().replace("+00:00", "Z"),
            end_time_utc=run_end_dt.isoformat().replace("+00:00", "Z"),
            duration_s=(run_end_dt - run_start_dt).total_seconds(),
            status=status,
        ),
        versions=_capture_versions(benchmark),
        hardware=_capture_hardware(benchmark),
        runtime=Runtime(
            startup_phase_times_s=StartupPhaseTimes(
                app_launch=app_launch_ms / 1000.0,
                env_creation=env_creation_ms / 1000.0,
                first_step=first_step_ms / 1000.0,
            ),
            iterations_completed=len(iter_times),
            total_wall_time_s=sum(iter_times),
            steps_per_iteration=steps_per_iter,
            iteration_time_s=_ms(iter_times),
            env_steps_per_s=_ms(env_steps_per_s_series),
            iterations_per_s=_ms(iters_per_s_series),
        ),
        resources=resources,
        learning=Learning(
            ema_alpha=args.ema_alpha,
            reward=LearningCurve(
                final_raw=reward_series[-1] if reward_series else 0.0,
                final_ema=_compute_ema(reward_series, args.ema_alpha),
                series_per_iter=None if args.no_series else reward_series,
            ),
            ep_length=LearningCurve(
                final_raw=ep_len_series[-1] if ep_len_series else 0.0,
                final_ema=_compute_ema(ep_len_series, args.ema_alpha),
                series_per_iter=None if args.no_series else ep_len_series,
            ),
        ),
    )
```

Copy `_capture_versions`, `_capture_hardware`, and `_synth_run_id` from Task 3 verbatim (or factor them into a new helper module `scripts/benchmarks/_schema_helpers.py` — your call; keeping copies for now avoids touching a third file).

Add `_capture_resources(benchmark)` that reads the `GPUInfoRecorder` / `CPUInfoRecorder` / `MemoryInfoRecorder` measurements (already populated by the monitoring loop in `BenchmarkMonitor`) and constructs a `Resources` dataclass with `{mean, std, peak}` fields. The recorders emit `StatisticalMeasurement` (which already has `mean`, `std`, `n`) and separate `SingleMeasurement(name="Peak X")` — pull both. If a metric isn't available (e.g., no GPU), fill zeros.

- [ ] **Step 5: Wire the emission into the end of `main()`**

At the top of `main()`, capture UTC start:
```python
from datetime import datetime, timezone
run_start_dt = datetime.now(timezone.utc)
```

Right after `benchmark._finalize_impl()` (inside the `if world_rank == 0:` block), add:
```python
if args_cli.schema_v1_output is not None:
    run_end_dt = datetime.now(timezone.utc)
    bundle = _build_training_bundle(
        log_data=log_data,
        rl_training_times=rl_training_times,
        agent_cfg=agent_cfg,
        env=env,
        args=args_cli,
        benchmark=benchmark,
        run_start_dt=run_start_dt,
        run_end_dt=run_end_dt,
        status="completed",
        app_launch_ms=(app_start_time_end - app_start_time_begin) / 1e6,
        env_creation_ms=Timer.get_timer_info("scene_creation") * 1000.0,
        first_step_ms=0.0,  # TODO: sample first-step time (see note below)
    )
    from isaaclab.test.benchmark.standard_schema import write_bundle_file
    write_bundle_file(bundle, args_cli.schema_v1_output)
```

First-step timing is not currently a named timer in this script. Wrap it: introduce a `Timer("first_step")` around the first `env.step` call inside `runner.learn`. Since `runner.learn` is not easily wrappable, instead record the first iteration's `rl_training_times["Collection Time"][0] + rl_training_times["Learning Time"][0]` as a proxy. Use that as `first_step_ms`.

- [ ] **Step 6: Wrap `main()` in try/finally for crash-path emission**

Same pattern as Task 3 Step 4: on exception, still write a `status="crashed"` bundle with whatever we captured. `log_data` may be empty or partial; `_build_training_bundle` must tolerate that (the numpy mean/std of empty lists currently returns 0.0 via our guard).

- [ ] **Step 7: Smoke-test standalone run (no v1 emission)**

Run:
```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
    --task Isaac-Ant-Direct-v0 --num_envs 64 --max_iterations 5 --headless
```
Expected: existing behavior — completes, emits the legacy backend output, no crashes.

- [ ] **Step 8: Smoke-test v1 emission**

Run:
```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
    --task Isaac-Ant-Direct-v0 --num_envs 64 --max_iterations 5 --headless \
    --backend physx --run_id test-training-smoke \
    --schema_v1_output /tmp/odin_smoke/training.json
```

Verify the JSON is well-formed and matches the schema:
```bash
./isaaclab.sh -p -c "
import json
d = json.load(open('/tmp/odin_smoke/training.json'))
assert d['schema_version'] == '1.0'
assert d['run']['framework'] == 'rsl_rl'
assert 'env_steps_per_s' in d['runtime']
assert 'final_ema' in d['learning']['reward']
print('OK', d['run']['run_id'], d['learning']['reward']['final_ema'])
"
```

- [ ] **Step 9: Format and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_rsl_rl.py
git commit -m "Emit v1.0 training bundle from benchmark_rsl_rl.py"
```

---

### Task 5: Create `benchmark_skrl.py`

Symmetric to `benchmark_rsl_rl.py`. Uses `scripts/reinforcement_learning/skrl/train.py` as the reference for how to configure and launch an SKRL trainer.

**Files:**
- Create: `scripts/benchmarks/benchmark_skrl.py`
- Read (for reference): `scripts/reinforcement_learning/skrl/train.py`

- [ ] **Step 1: Read the SKRL train script end-to-end**

Read `scripts/reinforcement_learning/skrl/train.py`. Identify:
1. How SKRL's `Trainer` is constructed (likely `SequentialTrainer` or `PPO` depending on config).
2. Where reward / episode length are logged (SKRL's `Agent` has a `.tracking_data` dict with keys like `Reward / Mean (iteration)`).
3. How seed / max_iterations / num_envs are passed.

- [ ] **Step 2: Copy `benchmark_rsl_rl.py` as scaffold**

```bash
cp scripts/benchmarks/benchmark_rsl_rl.py scripts/benchmarks/benchmark_skrl.py
```

Replace all `rsl_rl`/`rsl-rl` references with `skrl`. Replace imports:
- Drop: `from rsl_rl.runners import OnPolicyRunner`, `from isaaclab_rl.rsl_rl import ...`, `import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args`.
- Add: `from skrl.trainers.torch import SequentialTrainer` (or the equivalent your reference `skrl/train.py` uses), and corresponding agent imports.

Reuse the existing SKRL config loading via `resolve_task_config(args_cli.task, "skrl_cfg_entry_point")`.

- [ ] **Step 3: Replace `parse_tf_logs` consumption with SKRL's equivalent**

SKRL's trainer writes reward / ep_length to tensorboard under different tag names — typically `Reward / Total (iteration)` and `Episode / Length (iteration)` (verify by reading the SKRL source or running a small training and inspecting the event file with `tensorboard --inspect`). Map those to the `reward_series` / `ep_len_series` inputs of `_build_training_bundle`.

If the tags differ between SKRL versions, record the specific tag names used as a module-level constant so future breakage is obvious:

```python
_SKRL_REWARD_TAG = "Reward / Total (iteration)"
_SKRL_EP_LEN_TAG = "Episode / Length (iteration)"
```

- [ ] **Step 4: Smoke-test standalone run**

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_skrl.py \
    --task Isaac-Ant-Direct-v0 --num_envs 64 --max_iterations 5 --headless
```

If Ant doesn't have `skrl_cfg_entry_point` registered, this will error. Stop and log that for T2.1 gap tracking — don't try to fix it here.

- [ ] **Step 5: Smoke-test v1 emission**

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_skrl.py \
    --task Isaac-Ant-Direct-v0 --num_envs 64 --max_iterations 5 --headless \
    --backend physx --run_id test-skrl-smoke \
    --schema_v1_output /tmp/odin_smoke/training_skrl.json
```

Verify schema conformance:
```bash
./isaaclab.sh -p -c "
import json
d = json.load(open('/tmp/odin_smoke/training_skrl.json'))
assert d['run']['framework'] == 'skrl'
print('OK', d['run']['run_id'])
"
```

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_skrl.py
git commit -m "Add SKRL benchmark script emitting v1.0 training bundle"
```

---

### Task 6: User-facing docs page

**Files:**
- Create: `docs/source/features/benchmarking.md`

- [ ] **Step 1: Write a concise user-facing guide**

Content: what the three benchmark scripts do, how to invoke each (one command each with the key flags), the v1.0 schema quick reference (link to `standard_schema.py` for field-level docs), how to find the output, and a note that the full Odin bundle format lives under `docs/odin/` (link). Keep under ~150 lines. Include a note that `--backend` and `--run_id` are Odin integration flags and standalone users can ignore them.

- [ ] **Step 2: Generate docs build**

Run: `./isaaclab.sh -d`
Expected: docs build without errors. Any sphinx warnings on the new page should be addressed.

- [ ] **Step 3: Commit**

```bash
./isaaclab.sh -f
git add docs/source/features/benchmarking.md
git commit -m "Document IsaacLab benchmark scripts and v1.0 schema"
```

---

## Phase 3 — Odin runner wrappers

### Task 7: `tools/odin/common/manifest.py` + tests

Shared helpers used by both Hugin and Munin. Pure Python, no IsaacLab imports (so Odin-side tests are fast and don't need the Kit runtime).

**Files:**
- Create: `tools/odin/__init__.py` (empty)
- Create: `tools/odin/common/__init__.py` (empty)
- Create: `tools/odin/common/manifest.py`
- Create: `tools/odin/common/run_id.py`
- Create: `tools/odin/common/log_tail.py`
- Create: `tools/odin/tests/__init__.py` (empty)
- Create: `tools/odin/tests/test_run_id.py`
- Create: `tools/odin/tests/test_manifest.py`
- Create: `tools/odin/tests/test_log_tail.py`

- [ ] **Step 1: Write `run_id.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run ID format for Odin bundles.

Format: ``<framework>_<backend>_<task>_<date>_seed<seed>``
Example: ``rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42``
"""

from __future__ import annotations

from datetime import datetime, timezone

_FRAMEWORKS_PATH = {"rsl_rl": "rsl-rl", "skrl": "skrl"}
_FRAMEWORKS_JSON = {"rsl-rl": "rsl_rl", "skrl": "skrl"}


def compute_run_id(framework: str, backend: str, task: str, seed: int,
                   now: datetime | None = None) -> str:
    """Compute the canonical run_id for a new Odin run.

    Args:
        framework: Learning framework, e.g. ``"rsl_rl"`` or ``"skrl"``.
        backend: Physics backend, ``"physx"`` or ``"newton"``.
        task: Gym task ID, e.g. ``"Isaac-Ant-Direct-v0"``.
        seed: Integer seed.
        now: UTC datetime for the run-start timestamp. Defaults to
            :func:`datetime.now(timezone.utc)`.

    Returns:
        The run_id string.
    """
    if framework not in _FRAMEWORKS_PATH:
        raise ValueError(f"Unknown framework {framework!r}; expected one of {list(_FRAMEWORKS_PATH)}")
    if backend not in {"physx", "newton"}:
        raise ValueError(f"Unknown backend {backend!r}; expected 'physx' or 'newton'")
    if now is None:
        now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    fw_path = _FRAMEWORKS_PATH[framework]
    return f"{fw_path}_{backend}_{task}_{stamp}_seed{seed}"


def parse_run_id(run_id: str) -> dict:
    """Parse a run_id back into its components.

    Returns:
        Dict with keys ``framework`` (JSON form), ``backend``, ``task``, ``date``, ``seed``.

    Raises:
        ValueError: if the run_id is malformed.
    """
    parts = run_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"Malformed run_id {run_id!r}: expected at least 5 '_'-separated parts")
    fw_path = parts[0]
    backend = parts[1]
    seed_token = parts[-1]
    date = parts[-2]
    task = "_".join(parts[2:-2])  # task may contain underscores only if someone registered weird IDs
    if fw_path not in _FRAMEWORKS_JSON:
        raise ValueError(f"Unknown framework token {fw_path!r}")
    if not seed_token.startswith("seed"):
        raise ValueError(f"Expected seed token 'seed<int>', got {seed_token!r}")
    try:
        seed = int(seed_token[4:])
    except ValueError as e:
        raise ValueError(f"Invalid seed integer in {seed_token!r}") from e
    return {
        "framework": _FRAMEWORKS_JSON[fw_path],
        "backend": backend,
        "task": task,
        "date": date,
        "seed": seed,
    }
```

- [ ] **Step 2: Write `test_run_id.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Odin run_id format."""

from datetime import datetime, timezone

import pytest

from tools.odin.common.run_id import compute_run_id, parse_run_id


def test_compute_run_id_basic():
    now = datetime(2026, 4, 22, 13, 15, 0, tzinfo=timezone.utc)
    rid = compute_run_id("rsl_rl", "physx", "Isaac-Ant-Direct-v0", 42, now=now)
    assert rid == "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42"


def test_compute_run_id_rejects_unknown_framework():
    with pytest.raises(ValueError):
        compute_run_id("torch_rl", "physx", "Isaac-Ant-Direct-v0", 0)


def test_compute_run_id_rejects_unknown_backend():
    with pytest.raises(ValueError):
        compute_run_id("skrl", "mujoco", "Isaac-Ant-Direct-v0", 0)


def test_round_trip_ant():
    now = datetime(2026, 4, 22, 13, 15, 0, tzinfo=timezone.utc)
    rid = compute_run_id("skrl", "newton", "Isaac-Ant-Direct-v0", 7, now=now)
    parts = parse_run_id(rid)
    assert parts["framework"] == "skrl"
    assert parts["backend"] == "newton"
    assert parts["task"] == "Isaac-Ant-Direct-v0"
    assert parts["seed"] == 7


def test_parse_run_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_run_id("not_a_run_id")
```

- [ ] **Step 3: Run run_id tests to verify they pass**

From repo root:
```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_run_id.py -v
```
Expected: 5 tests PASS. If pytest can't import `tools.odin`, add an `__init__.py` path-style discovery: your `pyproject.toml` already allows `pytest tools/` style invocation.

If that fails, invoke via explicit PYTHONPATH:
```bash
PYTHONPATH=. ./isaaclab.sh -p -m pytest tools/odin/tests/test_run_id.py -v
```

- [ ] **Step 4: Write `log_tail.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

"""Log tail utility: capture the last N bytes of a stream to a file."""

from __future__ import annotations

from collections import deque

# Default tail size per the T1 spec: 16 KB per stream.
DEFAULT_TAIL_BYTES = 16 * 1024


def tail_bytes(data: bytes, max_bytes: int = DEFAULT_TAIL_BYTES) -> bytes:
    """Return the last ``max_bytes`` of ``data``, or all of it if shorter."""
    if len(data) <= max_bytes:
        return data
    return data[-max_bytes:]
```

- [ ] **Step 5: Write `test_log_tail.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

from tools.odin.common.log_tail import DEFAULT_TAIL_BYTES, tail_bytes


def test_tail_bytes_shorter_than_limit():
    assert tail_bytes(b"hello") == b"hello"


def test_tail_bytes_longer_than_limit():
    data = b"x" * (DEFAULT_TAIL_BYTES + 100)
    out = tail_bytes(data)
    assert len(out) == DEFAULT_TAIL_BYTES
    assert out == b"x" * DEFAULT_TAIL_BYTES


def test_tail_bytes_custom_limit():
    assert tail_bytes(b"abcdef", max_bytes=3) == b"def"
```

- [ ] **Step 6: Write `manifest.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

"""Write Odin manifest.json bundles.

Manifests are thin navigational indexes — see the T1 spec for the full schema.
"""

from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timezone

from isaaclab.test.benchmark.standard_schema import (
    Manifest,
    ManifestConfig,
    ManifestMachine,
    ManifestPhase,
    write_bundle_file,
)


def _get_git_info(repo_root: str) -> tuple[str | None, str | None]:
    """Return (commit, branch) or (None, None) if not a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return commit, branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def write_manifest(
    bundle_dir: str,
    run_id: str,
    framework: str,
    backend: str,
    task: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    run_start_dt: datetime,
    run_end_dt: datetime,
    startup_phase: ManifestPhase,
    training_phase: ManifestPhase,
    repo_root: str,
) -> str:
    """Write manifest.json to ``<bundle_dir>/manifest.json`` and return the path."""
    git_commit, git_branch = _get_git_info(repo_root)
    artifacts = sorted(os.listdir(bundle_dir)) if os.path.isdir(bundle_dir) else []
    manifest = Manifest(
        run_id=run_id,
        run_start_time_utc=run_start_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        run_end_time_utc=run_end_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        run_duration_s=(run_end_dt - run_start_dt).total_seconds(),
        config=ManifestConfig(framework=framework, backend=backend, task=task, seed=seed,
                              num_envs=num_envs, max_iterations=max_iterations),
        machine=ManifestMachine(hostname=socket.gethostname(),
                                git_commit=git_commit, git_branch=git_branch),
        phases={"startup": startup_phase, "training": training_phase},
        artifacts=artifacts,
    )
    path = os.path.join(bundle_dir, "manifest.json")
    write_bundle_file(manifest, path)
    return path
```

- [ ] **Step 7: Write `test_manifest.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from datetime import datetime, timezone

from isaaclab.test.benchmark.standard_schema import ManifestPhase
from tools.odin.common.manifest import write_manifest


def test_write_manifest_minimal(tmp_path):
    start = datetime(2026, 4, 22, 13, 15, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 22, 13, 47, 48, tzinfo=timezone.utc)
    path = write_manifest(
        bundle_dir=str(tmp_path),
        run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
        framework="rsl_rl", backend="physx", task="Isaac-Ant-Direct-v0", seed=42,
        num_envs=4096, max_iterations=500,
        run_start_dt=start, run_end_dt=end,
        startup_phase=ManifestPhase(file="startup.json", status="completed",
                                    duration_s=48.7, exit_code=0),
        training_phase=ManifestPhase(file="training.json", status="completed",
                                     duration_s=1942.1, exit_code=0),
        repo_root=str(tmp_path),  # not a git repo, so git info is None
    )
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["run_id"].startswith("rsl-rl_physx_")
    assert data["phases"]["training"]["exit_code"] == 0
    assert data["machine"]["git_commit"] is None  # tmp_path isn't a repo
```

- [ ] **Step 8: Run all Odin common tests**

```bash
PYTHONPATH=. ./isaaclab.sh -p -m pytest tools/odin/tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 9: Format and commit**

```bash
./isaaclab.sh -f
git add tools/odin/
git commit -m "Add Odin common helpers: run_id, manifest, log_tail"
```

---

### Task 8: `tools/odin/hugin/run.py` — RSL-RL runner wrapper

The runner wrapper. Takes CLI args, computes run_id, creates bundle dir, subprocess-launches startup + benchmark_rsl_rl, writes manifest.

**Files:**
- Create: `tools/odin/hugin/__init__.py` (empty)
- Create: `tools/odin/hugin/run.py`
- Create: `tools/odin/tests/test_hugin.py`

- [ ] **Step 1: Write `tools/odin/hugin/run.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

"""Hugin — Odin's RSL-RL runner wrapper.

One invocation = one Odin run. Subprocess-launches the IsaacLab startup
profiler and the IsaacLab RSL-RL benchmark script, collects their outputs
into a bundle directory, writes manifest.json, captures log tails on failure.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from isaaclab.test.benchmark.standard_schema import ManifestPhase
from tools.odin.common.log_tail import tail_bytes
from tools.odin.common.manifest import write_manifest
from tools.odin.common.run_id import compute_run_id

# Repo root — anchor used to locate the IsaacLab scripts.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_STARTUP_SCRIPT = os.path.join(_REPO_ROOT, "scripts/benchmarks/benchmark_startup.py")
_TRAINING_SCRIPT = os.path.join(_REPO_ROOT, "scripts/benchmarks/benchmark_rsl_rl.py")
_ISAACLAB_SH = os.path.join(_REPO_ROOT, "isaaclab.sh")


def _run_phase(
    cmd: list[str], bundle_dir: str, phase_name: str, output_json: str,
) -> ManifestPhase:
    """Run one subprocess phase; capture exit code, duration, and log tails on failure."""
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    start = datetime.now(timezone.utc)
    completed = subprocess.run(cmd, capture_output=True)
    end = datetime.now(timezone.utc)
    duration_s = (end - start).total_seconds()
    if completed.returncode != 0:
        status = "failed"
        with open(os.path.join(logs_dir, f"{phase_name}.stderr.log"), "wb") as f:
            f.write(tail_bytes(completed.stderr))
        with open(os.path.join(logs_dir, f"{phase_name}.stdout.log"), "wb") as f:
            f.write(tail_bytes(completed.stdout))
    else:
        status = "completed"
    return ManifestPhase(
        file=os.path.basename(output_json),
        status=status,
        duration_s=duration_s,
        exit_code=completed.returncode,
    )


def _copy_tb_events(training_log_dir: str, bundle_dir: str) -> None:
    """Best-effort copy of TB event files from RSL-RL's log dir into <bundle>/tb/."""
    tb_target = os.path.join(bundle_dir, "tb")
    os.makedirs(tb_target, exist_ok=True)
    for evt in glob.glob(os.path.join(training_log_dir, "**", "events.out.tfevents.*"),
                         recursive=True):
        try:
            shutil.copy2(evt, tb_target)
        except OSError:
            pass  # TB copy is best-effort; never block the run


def main():
    parser = argparse.ArgumentParser(description="Odin Hugin — RSL-RL runner wrapper.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", choices=["physx", "newton"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--runs_root", type=str, default="./odin_runs")
    parser.add_argument("--ema_alpha", type=float, default=0.05)
    parser.add_argument("--no_series", action="store_true", default=False)
    parser.add_argument("--skip_startup", action="store_true", default=False,
                        help="Skip the dense startup-profile subprocess (training-only run).")
    args = parser.parse_args()

    run_start = datetime.now(timezone.utc)
    run_id = compute_run_id("rsl_rl", args.backend, args.task, args.seed, now=run_start)
    bundle_dir = os.path.abspath(os.path.join(args.runs_root, run_id))
    os.makedirs(bundle_dir, exist_ok=True)

    startup_phase = ManifestPhase(file="startup.json", status="completed", duration_s=0.0, exit_code=0)
    if not args.skip_startup:
        startup_phase = _run_phase(
            cmd=[_ISAACLAB_SH, "-p", _STARTUP_SCRIPT,
                 "--task", args.task, "--num_envs", str(args.num_envs),
                 "--seed", str(args.seed), "--headless",
                 "--backend", args.backend, "--run_id", run_id,
                 "--schema_v1_output", os.path.join(bundle_dir, "startup.json")],
            bundle_dir=bundle_dir, phase_name="startup",
            output_json=os.path.join(bundle_dir, "startup.json"),
        )

    training_phase = _run_phase(
        cmd=[_ISAACLAB_SH, "-p", _TRAINING_SCRIPT,
             "--task", args.task, "--num_envs", str(args.num_envs),
             "--seed", str(args.seed), "--max_iterations", str(args.max_iterations),
             "--headless",
             "--backend", args.backend, "--run_id", run_id,
             "--schema_v1_output", os.path.join(bundle_dir, "training.json"),
             "--ema_alpha", str(args.ema_alpha)]
            + (["--no_series"] if args.no_series else []),
        bundle_dir=bundle_dir, phase_name="training",
        output_json=os.path.join(bundle_dir, "training.json"),
    )

    # Best-effort TB copy: look for the most recent rsl_rl log dir.
    rsl_rl_logs = os.path.join(_REPO_ROOT, "logs", "rsl_rl")
    if os.path.isdir(rsl_rl_logs):
        for experiment_dir in sorted(os.listdir(rsl_rl_logs), reverse=True):
            experiment_path = os.path.join(rsl_rl_logs, experiment_dir)
            if os.path.isdir(experiment_path):
                _copy_tb_events(experiment_path, bundle_dir)
                break

    run_end = datetime.now(timezone.utc)
    write_manifest(
        bundle_dir=bundle_dir,
        run_id=run_id,
        framework="rsl_rl",
        backend=args.backend,
        task=args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        run_start_dt=run_start,
        run_end_dt=run_end,
        startup_phase=startup_phase,
        training_phase=training_phase,
        repo_root=_REPO_ROOT,
    )

    # Exit non-zero if any phase failed so the dispatcher (T3) can detect it.
    if startup_phase.exit_code != 0 or training_phase.exit_code != 0:
        sys.exit(max(startup_phase.exit_code, training_phase.exit_code))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `test_hugin.py` with fake subprocesses**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for Hugin: mocks subprocess.run to avoid launching real trainings."""

import json
import os
from unittest import mock

import pytest

from tools.odin.hugin import run as hugin_run


def _fake_run_factory(bundle_dir: str):
    """Return a subprocess.run stub that pretends to write startup.json/training.json."""
    def _fake_run(cmd, capture_output=True):
        # Locate the --schema_v1_output path in the command and fake-write it.
        out_idx = cmd.index("--schema_v1_output") + 1
        out_path = cmd[out_idx]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write('{"schema_version": "1.0", "fake": true}\n')
        class R:
            returncode = 0
            stdout = b"fake stdout"
            stderr = b"fake stderr"
        return R()
    return _fake_run


def test_hugin_happy_path(tmp_path, monkeypatch):
    bundle_root = str(tmp_path)
    monkeypatch.setattr(hugin_run.subprocess, "run", _fake_run_factory(bundle_root))
    monkeypatch.setattr("sys.argv", [
        "hugin", "--task", "Isaac-Ant-Direct-v0", "--backend", "physx",
        "--seed", "42", "--num_envs", "64", "--max_iterations", "5",
        "--runs_root", bundle_root,
    ])
    hugin_run.main()

    # Find the created bundle dir
    bundles = [d for d in os.listdir(bundle_root) if d.startswith("rsl-rl_physx_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    assert os.path.exists(os.path.join(bundle, "startup.json"))
    assert os.path.exists(os.path.join(bundle, "training.json"))
    assert os.path.exists(os.path.join(bundle, "manifest.json"))
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["exit_code"] == 0


def test_hugin_failure_path_writes_logs(tmp_path, monkeypatch):
    def _failing_run(cmd, capture_output=True):
        out_idx = cmd.index("--schema_v1_output") + 1
        out_path = cmd[out_idx]
        # Don't write the output file — simulate a crash before emission.
        class R:
            returncode = 7
            stdout = b"partial stdout"
            stderr = b"traceback..."
        return R()

    bundle_root = str(tmp_path)
    monkeypatch.setattr(hugin_run.subprocess, "run", _failing_run)
    monkeypatch.setattr("sys.argv", [
        "hugin", "--task", "Isaac-Ant-Direct-v0", "--backend", "newton",
        "--seed", "1", "--runs_root", bundle_root, "--skip_startup",
    ])
    with pytest.raises(SystemExit) as exc:
        hugin_run.main()
    assert exc.value.code == 7

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("rsl-rl_newton_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    # training log tail should be present
    assert os.path.exists(os.path.join(bundle, "logs", "training.stderr.log"))
    with open(os.path.join(bundle, "logs", "training.stderr.log"), "rb") as f:
        assert f.read() == b"traceback..."
    # manifest still written
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["status"] == "failed"
    assert m["phases"]["training"]["exit_code"] == 7
```

- [ ] **Step 3: Run hugin tests**

```bash
PYTHONPATH=. ./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 4: Format and commit**

```bash
./isaaclab.sh -f
git add tools/odin/hugin/ tools/odin/tests/test_hugin.py
git commit -m "Add Odin Hugin RSL-RL runner wrapper"
```

---

### Task 9: `tools/odin/munin/run.py` — SKRL runner wrapper

Near-verbatim copy of Hugin with the training script swapped.

**Files:**
- Create: `tools/odin/munin/__init__.py` (empty)
- Create: `tools/odin/munin/run.py`
- Create: `tools/odin/tests/test_munin.py`

- [ ] **Step 1: Copy Hugin and adapt**

```bash
mkdir -p tools/odin/munin
cp tools/odin/hugin/run.py tools/odin/munin/run.py
touch tools/odin/munin/__init__.py
```

In `tools/odin/munin/run.py`: replace `benchmark_rsl_rl.py` → `benchmark_skrl.py`, `"rsl_rl"` → `"skrl"`, `rsl_rl` log dir lookup → `skrl` log dir lookup (`os.path.join(_REPO_ROOT, "logs", "skrl")`), and `compute_run_id("rsl_rl", ...)` → `compute_run_id("skrl", ...)`.

Update the description string and module docstring.

- [ ] **Step 2: Copy test_hugin.py and adapt to test_munin.py**

Replace `hugin` / `rsl-rl_` / `rsl_rl` with `munin` / `skrl_` / `skrl`.

- [ ] **Step 3: Run Munin tests**

```bash
PYTHONPATH=. ./isaaclab.sh -p -m pytest tools/odin/tests/test_munin.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 4: Format and commit**

```bash
./isaaclab.sh -f
git add tools/odin/munin/ tools/odin/tests/test_munin.py
git commit -m "Add Odin Munin SKRL runner wrapper"
```

---

### Task 10: Odin README + `.gitignore`

**Files:**
- Create: `tools/odin/README.md`
- Create: `tools/odin/.gitignore`

- [ ] **Step 1: Write README**

```markdown
# Odin — Evaluation Harness (In-Tree)

Codename for the multi-backend IsaacLab evaluation harness. See the
[living architecture reference](../../docs/odin/architecture.md) for the
cross-task overview.

This directory currently lives inside IsaacLab for development convenience.
When Odin graduates, this whole directory moves to its own repo; the
IsaacLab-side benchmark scripts (`scripts/benchmarks/benchmark_*.py` and
`source/isaaclab/isaaclab/test/benchmark/standard_schema.py`) stay in place
and remain independently usable.

## Components

- `hugin/` — RSL-RL benchmark runner wrapper.
- `munin/` — SKRL benchmark runner wrapper.
- `common/` — shared helpers (run_id format, manifest writer, log tail).
- `tests/` — unit and integration tests (run without Kit runtime).

## Running a single bundle locally

```bash
# RSL-RL on PhysX
PYTHONPATH=. ./isaaclab.sh -p tools/odin/hugin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 4096 --max_iterations 500

# SKRL on Newton
PYTHONPATH=. ./isaaclab.sh -p tools/odin/munin/run.py \
    --task Isaac-Ant-Direct-v0 --backend newton --seed 42 \
    --num_envs 4096 --max_iterations 500
```

Outputs land under `./odin_runs/<run_id>/` by default. See
[the spec](../../docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md)
for the bundle layout and schema.
```

- [ ] **Step 2: Write `.gitignore`**

```
odin_runs/
```

- [ ] **Step 3: Commit**

```bash
git add tools/odin/README.md tools/odin/.gitignore
git commit -m "Add Odin in-tree README and runs ignore list"
```

---

## Phase 4 — Dry-run + reference bundles

### Task 11: Four-bundle dry run on `Isaac-Ant-Direct-v0`

Produces the four reference bundles the spec promises. If any cell (framework, backend) is blocked by a missing config or backend gap, record it as T2.1 input and commit whichever succeeded.

**Execution notes:**
- Run sequentially. Never two at once (GPU parallel runs cause segfaults — see memory `feedback_gpu_tests.md`).
- Keep `num_envs=4096`, `max_iterations=500`, `seed=42` — matches `run_training_benchmarks.sh` for comparability.
- Expected runtime per bundle: ~30 minutes on H100, less on smaller GPUs.

- [ ] **Step 1: Run RSL-RL × PhysX**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/hugin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 4096 --max_iterations 500 \
    --runs_root ./odin_runs
```

Verify the bundle:
```bash
ls ./odin_runs/rsl-rl_physx_Isaac-Ant-Direct-v0_*/
./isaaclab.sh -p -c "
import json, glob
bundle = glob.glob('./odin_runs/rsl-rl_physx_*')[0]
m = json.load(open(bundle + '/manifest.json'))
assert m['phases']['training']['status'] == 'completed'
print('OK', bundle)
"
```

- [ ] **Step 2: Run RSL-RL × Newton**

Same command with `--backend newton`. If Newton doesn't support Ant, record the error in a notes file:

```bash
echo "RSL-RL × Newton on Isaac-Ant-Direct-v0 failed: <paste error tail>" \
    >> ./odin_runs/_DRY_RUN_NOTES.md
```

and move on.

- [ ] **Step 3: Run SKRL × PhysX**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/munin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 4096 --max_iterations 500 \
    --runs_root ./odin_runs
```

If `skrl_cfg_entry_point` is not registered for Ant, record the gap:

```bash
echo "SKRL × PhysX on Isaac-Ant-Direct-v0 failed: no skrl_cfg_entry_point registered." \
    >> ./odin_runs/_DRY_RUN_NOTES.md
```

- [ ] **Step 4: Run SKRL × Newton**

Same as Step 3 with `--backend newton`. Record gaps same way.

- [ ] **Step 5: Copy successful bundles into the committed reference set**

```bash
mkdir -p docs/odin/reference_runs
for bundle in ./odin_runs/rsl-rl_physx_* ./odin_runs/rsl-rl_newton_* \
              ./odin_runs/skrl_physx_* ./odin_runs/skrl_newton_*; do
    [ -d "$bundle" ] || continue
    target="docs/odin/reference_runs/$(basename $bundle)"
    mkdir -p "$target"
    cp "$bundle/manifest.json" "$bundle/training.json" "$bundle/startup.json" "$target/" 2>/dev/null || true
    cat > "$target/README.md" <<EOF
Reference bundle for Odin T1. TB events and logs/ are omitted here (too large /
noisy for version control); see the full bundle on the producing machine.

- Task: Isaac-Ant-Direct-v0
- num_envs: 4096, max_iterations: 500, seed: 42
EOF
done
```

- [ ] **Step 6: Move the dry-run notes into the repo**

If `_DRY_RUN_NOTES.md` was created, copy it to the reference runs folder as input for T2.1:

```bash
[ -f ./odin_runs/_DRY_RUN_NOTES.md ] && \
    cp ./odin_runs/_DRY_RUN_NOTES.md docs/odin/reference_runs/_DRY_RUN_NOTES.md
```

- [ ] **Step 7: Commit the reference set**

```bash
git add docs/odin/reference_runs/
git commit -m "Add Odin T1 reference bundles for Isaac-Ant-Direct-v0"
```

---

### Task 12: Schema-validation test against committed bundles

**Files:**
- Create: `source/isaaclab/test/benchmark/test_reference_bundles.py`

- [ ] **Step 1: Write the validation test**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (...).
# SPDX-License-Identifier: BSD-3-Clause

"""Validate committed Odin reference bundles against the v1.0 schema."""

import json
import os
from pathlib import Path

import pytest

from isaaclab.test.benchmark.standard_schema import SCHEMA_VERSION

_REFERENCE_ROOT = Path(__file__).resolve().parents[4] / "docs" / "odin" / "reference_runs"


def _iter_bundles():
    if not _REFERENCE_ROOT.exists():
        return
    for entry in sorted(_REFERENCE_ROOT.iterdir()):
        if entry.is_dir() and (entry / "manifest.json").exists():
            yield entry


@pytest.mark.parametrize("bundle", list(_iter_bundles()),
                         ids=lambda b: b.name if b is not None else "none")
def test_bundle_schema_v1(bundle):
    manifest = json.load(open(bundle / "manifest.json"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    for key in ("run_id", "config", "machine", "phases", "artifacts"):
        assert key in manifest, f"{bundle.name}: missing manifest key {key}"

    training = json.load(open(bundle / "training.json"))
    assert training["schema_version"] == SCHEMA_VERSION
    for key in ("run", "versions", "hardware", "runtime", "resources", "learning"):
        assert key in training, f"{bundle.name}: missing training key {key}"
    assert "env_steps_per_s" in training["runtime"]
    assert "final_ema" in training["learning"]["reward"]

    startup = json.load(open(bundle / "startup.json"))
    assert startup["schema_version"] == SCHEMA_VERSION
    for key in ("run", "versions", "hardware", "phases", "config"):
        assert key in startup, f"{bundle.name}: missing startup key {key}"
```

- [ ] **Step 2: Run the test**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_reference_bundles.py -v
```
Expected: one test PASS per committed bundle (as many as Task 11 produced). If zero bundles exist, the parametrize list is empty and pytest skips — that's fine but stop and investigate why no bundles landed.

- [ ] **Step 3: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/test/benchmark/test_reference_bundles.py
git commit -m "Validate Odin reference bundles against v1.0 schema"
```

---

## Phase 5 — Closing

### Task 13: CHANGELOG + extension.toml bumps

Per `AGENTS.md`: any change under `source/<package>/` must bump that package's `CHANGELOG.rst` and `config/extension.toml` version.

**Files:**
- Modify: `source/isaaclab/docs/CHANGELOG.rst`
- Modify: `source/isaaclab/config/extension.toml`

- [ ] **Step 1: Determine the new isaaclab version**

Read the current version from `source/isaaclab/config/extension.toml` (e.g. `4.6.8`) and bump the patch: `4.6.9`.

- [ ] **Step 2: Add the changelog entry at the top**

Add a new version heading (tilde-underline) and `Added` section — do NOT modify existing entries.

```rst
4.6.9 (2026-04-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab.test.benchmark.standard_schema` defining the v1.0 Odin
  benchmark bundle format (training, startup, manifest JSON schemas).
* Extended :mod:`scripts.benchmarks.benchmark_startup` and
  :mod:`scripts.benchmarks.benchmark_rsl_rl` to emit v1.0 ``startup.json``
  and ``training.json`` via a new ``--schema_v1_output`` CLI flag, with
  reward/episode-length EMA smoothing for stable final-value reporting.
* Added :mod:`scripts.benchmarks.benchmark_skrl` mirroring the RSL-RL
  benchmark script for SKRL trainings with the same v1.0 output schema.
```

- [ ] **Step 3: Bump `config/extension.toml`**

Change `version = "4.6.8"` to `version = "4.6.9"` (or whatever matches).

- [ ] **Step 4: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/docs/CHANGELOG.rst source/isaaclab/config/extension.toml
git commit -m "Bump isaaclab to 4.6.9 for Odin T1 benchmark additions"
```

---

### Task 14: Update the architecture reference with T1 status

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 1: Update the task table in §6**

Change `T1` row from `🟡 (design approved, impl pending)` to `✅`.

- [ ] **Step 2: Update the glossary with any component names confirmed by impl**

Confirm the `Hugin → RSL-RL` / `Munin → SKRL` mapping now that it's code, not just plan.

- [ ] **Step 3: Add a change log entry in §9**

```markdown
| 2026-04-22 | T1 implementation complete: v1.0 schema, three benchmark scripts upgraded/added, Hugin + Munin runners, four reference bundles on Ant Direct committed. | Odin T1 |
```

- [ ] **Step 4: Note startup phases covered**

In §3 or §4, add a note that `startup.json` captures five phases (`app_launch`, `python_imports`, `task_config`, `env_creation`, `first_step`) — the T1 spec originally listed three; the implementation reused `benchmark_startup.py`'s richer existing split.

- [ ] **Step 5: Commit**

```bash
git add docs/odin/architecture.md
git commit -m "Mark Odin T1 complete in architecture reference"
```

---

### Task 15: Final verification sweep

- [ ] **Step 1: Full pre-commit pass**

```bash
./isaaclab.sh -f
```
Expected: no pending modifications.

- [ ] **Step 2: All new tests green**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_standard_schema.py source/isaaclab/test/benchmark/test_reference_bundles.py -v
PYTHONPATH=. ./isaaclab.sh -p -m pytest tools/odin/tests/ -v
```
Expected: all tests PASS (sequential, not parallel).

- [ ] **Step 3: Smoke-verify the full chain end-to-end**

Re-run one Odin bundle via Hugin with `--max_iterations 5` to make sure the whole pipeline works after all the commits:

```bash
rm -rf /tmp/odin_smoke_final
PYTHONPATH=. ./isaaclab.sh -p tools/odin/hugin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 64 --max_iterations 5 \
    --runs_root /tmp/odin_smoke_final
ls /tmp/odin_smoke_final/rsl-rl_physx_Isaac-Ant-Direct-v0_*/
```
Expected: bundle dir exists with `manifest.json`, `training.json`, `startup.json`.

- [ ] **Step 4: Review the committed reference bundles one more time**

```bash
git diff HEAD~12 -- docs/odin/reference_runs/
```
Expected: only the reference bundles, no unexpected files.

- [ ] **Step 5: Summary commit message if anything cleanup-worthy**

If the previous 14 commits left any loose ends (stray debug prints, orphaned TODOs), clean up now in one final focused commit.

---

## Self-review checklist (run before handing back)

- [ ] Spec coverage: every spec section has at least one task.
  - Architecture → Tasks 1, 3, 4, 5, 7, 8, 9
  - Schema → Tasks 1, 2, 12
  - Naming + bundle layout → Tasks 7, 8, 9
  - Dry-run deliverable → Task 11
  - Upstream vs Odin split → Task 13 + file locations throughout
  - Testing → Tasks 2, 7, 8, 9, 12, 15
- [ ] Placeholder scan: no "TBD" / "add appropriate error handling" / "similar to Task N" in live steps. Surviving `TODO` comments are explicit and limited (only in schema helpers where a follow-up is called out).
- [ ] Type consistency: `TrainingBundle` / `StartupBundle` / `Manifest` names match between `standard_schema.py`, benchmark scripts, and `manifest.py`. `compute_run_id` and `parse_run_id` use the same framework token set.
- [ ] Commit discipline: every task ends with a commit; commit messages follow `AGENTS.md` style.

---

## Known execution-time risks (heads-up for the implementer)

1. **`scripts/benchmarks/utils.py::parse_cprofile_stats`** currently returns `(label, tottime_ms, cumtime_ms)` tuples — no call counts. Schema's `CProfileFunction.calls` is filled with `0` for v1. If calls are easy to extract, do it in the same commit as Task 3.

2. **Newton support for Ant** is unverified. Task 11 expects failures to be possible and records them — don't treat a failed bundle as a task-level blocker.

3. **SKRL tensorboard tag names** differ from RSL-RL. Task 5 Step 3 calls this out explicitly — verify against the actual SKRL training output, not assumptions.

4. **`BenchmarkMonitor` resource sampling cadence** defaults to 1 s (`interval=1.0` in `benchmark_rsl_rl.py:236`). For very short test runs (5 iterations) the resource stats may be noisy or empty. Reference bundles use 500 iterations so this only matters for smoke tests.

5. **Git info** in `manifest.py::_get_git_info` shells out to `git`. When running inside a Docker container on a Valkyrie (T3), the repo must be mounted and `git` must be on PATH. T1 doesn't exercise this path but Task 7 keeps the fallback (`None, None`) explicit.
