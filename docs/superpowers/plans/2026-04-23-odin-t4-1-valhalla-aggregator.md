# Odin T4.1 Valhalla Aggregator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `odin_runs/<dispatch_id>/aggregate.json` generation — a per-dispatch rollup of bundle metrics, nested by `(task, framework, backend)` with per-seed drill-down + cross-seed aggregates, produced both automatically at the end of `run_dispatch` and manually via a new `odin-aggregate` CLI.

**Architecture:** New `tools/odin/valhalla/` package with five modules: `stats` (reusable mean/std/cv + z-score divergence), `aggregator` (core read-and-roll logic), `writer` (atomic JSON write), `cli` (argparse wrapper), and `__init__` (public exports). T3.1's `run_dispatch` gets an auto-call at the end plus a `--skip-aggregate` opt-out. No upstream IsaacLab changes.

**Tech Stack:** Python 3.10+, standard library only (`json`, `dataclasses`, `pathlib`, `tempfile`, `os`, `argparse`), pytest.

**Spec:** `docs/superpowers/specs/2026-04-23-odin-t4-1-valhalla-aggregator-design.md`

---

## File Structure

**New files:**
- `tools/odin/valhalla/__init__.py` — public re-exports (`aggregate_dispatch`, `AggregateOptions`, `write_aggregate`).
- `tools/odin/valhalla/stats.py` — `Stats` dataclass, `stats_over(values)`, `is_divergent(values, z)`.
- `tools/odin/valhalla/aggregator.py` — `AggregateOptions`, `aggregate_dispatch(dispatch_dir, options)`.
- `tools/odin/valhalla/writer.py` — `write_aggregate(dispatch_dir, aggregate, *, overwrite)`.
- `tools/odin/valhalla/cli.py` — `main()` / argparse / `LATEST` resolver.
- `tools/odin/tests/test_valhalla_stats.py`
- `tools/odin/tests/test_valhalla_aggregator.py`
- `tools/odin/tests/test_valhalla_writer.py`
- `tools/odin/tests/test_valhalla_cli.py`
- `tools/odin/tests/test_valhalla_integration.py`

**Modified:**
- `tools/odin/asgard/runner.py` — call `aggregate_dispatch` + `write_aggregate` at the end of `run_dispatch` (skippable via option).
- `tools/odin/asgard/cli.py` — new `--skip-aggregate` passthrough flag.

**Unchanged (confirmed):**
- Bundle shape: `manifest.json`, `training.json`, `startup.json` v1.0 schemas (T4.1 is a pure consumer).
- `dispatch.json` schema (T4.1 reads from it; no writes).
- T1 benchmark scripts + Hugin / Munin runners.

**Task ordering rationale:** Tasks 1-5 build the Valhalla package bottom-up — stats primitives first, then the aggregator that uses them, then writer (independent of aggregator), then CLI (depends on both), then package exports. Task 6 wires the auto-call into T3.1. Task 7 is the cross-subsystem integration test. Task 8 is the sweep (full suite + pre-commit).

---

### Task 1: `stats.py` — shared mean/std/min/max/cv_pct + divergence detector

**Files:**
- Create: `tools/odin/valhalla/stats.py`
- Create: `tools/odin/tests/test_valhalla_stats.py`

- [ ] **Step 1: Write the failing test file**

Create `tools/odin/tests/test_valhalla_stats.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.stats — mean / std / cv_pct + divergence helper."""

from __future__ import annotations

import math

import pytest

from tools.odin.valhalla.stats import Stats, is_divergent, stats_over


def test_stats_over_n_one_has_zero_std_and_cv():
    s = stats_over([5.0])
    assert s == Stats(mean=5.0, std=0.0, min=5.0, max=5.0, cv_pct=0.0)


def test_stats_over_n_two_uses_population_std():
    # Population std of [4, 6] = sqrt(((-1)^2 + 1^2) / 2) = 1.0
    s = stats_over([4.0, 6.0])
    assert s.mean == 5.0
    assert s.std == pytest.approx(1.0)
    assert s.min == 4.0
    assert s.max == 6.0
    assert s.cv_pct == pytest.approx(20.0)


def test_stats_over_n_three_uses_sample_std():
    # Sample std (ddof=1) of [1, 2, 3] = sqrt(((-1)^2 + 0 + 1^2) / 2) = 1.0
    s = stats_over([1.0, 2.0, 3.0])
    assert s.mean == 2.0
    assert s.std == pytest.approx(1.0)
    assert s.cv_pct == pytest.approx(50.0)


def test_stats_over_mean_zero_yields_zero_cv_not_nan():
    s = stats_over([-1.0, 0.0, 1.0])
    assert s.mean == 0.0
    assert s.cv_pct == 0.0
    assert not math.isnan(s.cv_pct)


def test_stats_over_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        stats_over([])


def test_is_divergent_below_threshold_returns_empty():
    # [1, 1, 1, 1, 1.05] — 1.05 is within 2*std of mean.
    assert is_divergent([1.0, 1.0, 1.0, 1.0, 1.05], z=2.0) == []


def test_is_divergent_outlier_flags_its_index():
    # [1, 1, 1, 1, 10] — 10 is clearly > 2*std from mean.
    assert is_divergent([1.0, 1.0, 1.0, 1.0, 10.0], z=2.0) == [4]


def test_is_divergent_higher_z_may_skip_outlier():
    # With z=3.0 the single outlier might not breach, depends on numbers.
    # Use a case that's 2-sigma (flags at z=2) but not 3-sigma (skipped at z=3).
    values = [1.0, 1.0, 1.0, 1.0, 3.0]
    out_at_2 = is_divergent(values, z=2.0)
    out_at_3 = is_divergent(values, z=3.0)
    assert len(out_at_2) >= 1  # flags the outlier
    assert out_at_3 == []  # skips at higher threshold


def test_is_divergent_n_less_than_three_returns_empty():
    assert is_divergent([1.0], z=2.0) == []
    assert is_divergent([1.0, 100.0], z=2.0) == []
```

- [ ] **Step 2: Run test to verify it fails with ImportError**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_stats.py -v --confcutdir=tools/odin
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tools.odin.valhalla'`.

- [ ] **Step 3: Implement `stats.py`**

Create `tools/odin/valhalla/stats.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stats helpers for the Valhalla aggregator.

Exposes :class:`Stats` (the mean/std/min/max/cv_pct blob used in
``aggregate.json`` rows) plus :func:`stats_over` to compute one over a
list of samples, and :func:`is_divergent` to flag outlier seeds via a
z-score threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Stats", "stats_over", "is_divergent"]


@dataclass
class Stats:
    """Cross-seed stats blob emitted inside each row's ``aggregate`` block."""

    mean: float
    std: float
    min: float
    max: float
    cv_pct: float


def stats_over(values: list[float]) -> Stats:
    """Compute the aggregate stats over ``values``.

    Args:
        values: Completed-seed samples. Must be non-empty.

    Returns:
        :class:`Stats` with population std for ``len(values) == 2`` and
        sample std (ddof=1) for ``len(values) >= 3``. ``len(values) == 1``
        yields ``std=0.0``, ``cv_pct=0.0``.

    Raises:
        ValueError: When ``values`` is empty.
    """
    if not values:
        raise ValueError("stats_over requires at least one sample, got empty list")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    elif n == 2:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)  # population, ddof=0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))  # sample, ddof=1
    cv_pct = 0.0 if mean == 0.0 else 100.0 * std / abs(mean)
    return Stats(mean=mean, std=std, min=min(values), max=max(values), cv_pct=cv_pct)


def is_divergent(values: list[float], z: float) -> list[int]:
    """Return the indices of values farther than ``z`` standard deviations from the mean.

    Uses sample std (ddof=1). Returns an empty list for ``len(values) < 3``
    — two-sample outlier detection is meaningless.

    Args:
        values: Per-seed metric samples.
        z: Threshold in standard-deviation multiples.

    Returns:
        Sorted list of indices of offending samples (empty if none).
    """
    if len(values) < 3:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    if std == 0.0:
        return []
    return [i for i, v in enumerate(values) if abs(v - mean) > z * std]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_stats.py -v --confcutdir=tools/odin
```

Expected: 9 PASS.

- [ ] **Step 5: Run pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/stats.py tools/odin/tests/test_valhalla_stats.py
git commit -m "Add Valhalla stats helpers (mean/std/cv + divergence)"
```

Subject is 49 chars — under 50.

---

### Task 2: `aggregator.py` — read bundles, classify seeds, build dict

**Files:**
- Create: `tools/odin/valhalla/aggregator.py`
- Create: `tools/odin/tests/test_valhalla_aggregator.py`

- [ ] **Step 1: Write the failing test — happy path only**

Create `tools/odin/tests/test_valhalla_aggregator.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.aggregator on synthetic dispatch directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.aggregator import AggregateOptions, aggregate_dispatch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _write_completed_bundle(
    dispatch_dir: Path,
    run_id: str,
    *,
    reward_final_ema: float,
    ep_length_final_ema: float = 900.0,
    iter_time_mean: float = 0.5,
    iter_time_std: float = 0.05,
    env_steps_per_s_mean: float = 250000.0,
    ram_gb_peak: float = 8.0,
    gpu_mem_gb_peak: float = 4.0,
    commit_sha: str = "abc123",
    hostname: str = "valkyrie-01.internal",
) -> Path:
    bundle = dispatch_dir / run_id
    _write_json(
        bundle / "manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "machine": {"hostname": hostname, "git_commit": commit_sha, "git_branch": "main"},
            "phases": {
                "startup": {"file": "startup.json", "status": "completed", "duration_s": 30.0, "exit_code": 0},
                "training": {"file": "training.json", "status": "completed", "duration_s": 150.0, "exit_code": 0},
            },
            "config": {"framework": "rsl_rl", "backend": "physx", "task": "Isaac-Ant-Direct-v0", "seed": 42, "num_envs": 4096, "max_iterations": 300},
            "run_start_time_utc": "2026-04-23T10:00:00Z",
            "run_end_time_utc": "2026-04-23T10:03:00Z",
            "run_duration_s": 180.0,
            "artifacts": ["logs", "startup.json", "training.json", "training_data"],
        },
    )
    _write_json(
        bundle / "training.json",
        {
            "schema_version": "1.0",
            "runtime": {
                "iterations_completed": 300,
                "total_wall_time_s": 150.0,
                "iteration_time_s": {"mean": iter_time_mean, "std": iter_time_std},
                "env_steps_per_s": {"mean": env_steps_per_s_mean, "std": env_steps_per_s_mean * 0.01},
                "iterations_per_s": {"mean": 1.0 / iter_time_mean, "std": 0.01},
                "startup_phase_times_s": {"app_launch": 4.5, "env_creation": 12.4, "first_step": 0.006},
            },
            "resources": {
                "ram_gb": {"mean": ram_gb_peak * 0.9, "peak": ram_gb_peak},
                "gpu_mem_gb": {"mean": gpu_mem_gb_peak * 0.9, "peak": gpu_mem_gb_peak},
            },
            "learning": {
                "reward": {"final_raw": reward_final_ema * 1.01, "final_ema": reward_final_ema, "series_per_iter": [0.0] * 300},
                "ep_length": {"final_raw": ep_length_final_ema * 1.02, "final_ema": ep_length_final_ema, "series_per_iter": [0.0] * 300},
            },
        },
    )
    return bundle


def _make_dispatch_json(dispatch_dir: Path, jobs: list[dict]) -> None:
    _write_json(
        dispatch_dir / "dispatch.json",
        {
            "schema_version": "1.0",
            "dispatch_id": dispatch_dir.name,
            "started_at": "2026-04-23T09:59:00Z",
            "ended_at": "2026-04-23T10:10:00Z",
            "seeds": [42, 43],
            "commit_sha": "abc123",
            "fleet": [{"host": "valkyrie-01.internal", "status": "idle", "current_run_id": None, "last_error": None}],
            "jobs": jobs,
        },
    )


def _job(
    run_id: str,
    *,
    task: str = "Isaac-Ant-Direct-v0",
    framework: str = "rsl_rl",
    backend: str = "physx",
    seed: int = 42,
    status: str = "completed",
    assigned_to: str = "valkyrie-01.internal",
    failure: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "task_id": task,
        "framework": framework,
        "backend": backend,
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": seed,
        "bundle_dir_name": run_id,
        "status": status,
        "assigned_to": assigned_to,
        "attempts": 1,
        "failure": failure,
        "preferred_not": [],
        "started_at": "2026-04-23T10:00:00Z",
        "ended_at": "2026-04-23T10:03:00Z",
    }


def test_happy_path_two_completed_seeds(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run43", reward_final_ema=110.0)
    _make_dispatch_json(
        dispatch,
        [_job("run42", seed=42), _job("run43", seed=43)],
    )
    agg = aggregate_dispatch(dispatch)

    assert agg["schema_version"] == "1.0"
    assert agg["dispatch_id"] == "20260423-100000"
    assert agg["commit_sha"] == "abc123"
    assert agg["totals"] == {"tasks": 1, "runs": 2, "completed": 2, "failed": 0}
    assert len(agg["rows"]) == 1
    row = agg["rows"][0]
    assert row["task"] == "Isaac-Ant-Direct-v0"
    assert row["framework"] == "rsl_rl"
    assert row["backend"] == "physx"
    assert set(row["seeds"].keys()) == {"42", "43"}
    assert row["seeds"]["42"]["reward_final_ema"] == 100.0
    assert row["seeds"]["43"]["reward_final_ema"] == 110.0
    assert row["seeds"]["42"]["status"] == "completed"
    assert row["aggregate"]["n_seeds_completed"] == 2
    assert row["aggregate"]["n_seeds_failed"] == 0
    assert row["aggregate"]["reward_final_ema"]["mean"] == pytest.approx(105.0)
    assert row["divergent_seeds"] == []
    assert agg["failures"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_aggregator.py -v --confcutdir=tools/odin
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tools.odin.valhalla.aggregator'`.

- [ ] **Step 3: Implement minimal `aggregator.py`**

Create `tools/odin/valhalla/aggregator.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core Valhalla aggregator — rolls bundle JSONs into a single dispatch aggregate.

Input: an ``odin_runs/<dispatch_id>/`` directory produced by T3.1's
dispatcher. The directory holds one ``dispatch.json`` (source of truth
for which jobs existed) plus one ``<run_id>/`` subdirectory per job.

Output: a plain Python ``dict`` matching the T4.1 ``aggregate.json``
schema v1.0. The dict is returned — writing it to disk is
:func:`valhalla.writer.write_aggregate`'s job.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.odin.valhalla.stats import is_divergent, stats_over

__all__ = ["AggregateOptions", "aggregate_dispatch"]

SCHEMA_VERSION = "1.0"

# Metrics aggregated into Stats blobs. Each entry is
# (per-seed field name, training.json dotted path).
_AGGREGATED_METRICS: list[tuple[str, str]] = [
    ("reward_final_ema",      "learning.reward.final_ema"),
    ("ep_length_final_ema",   "learning.ep_length.final_ema"),
    ("iter_time_s_mean",      "runtime.iteration_time_s.mean"),
    ("env_steps_per_s_mean",  "runtime.env_steps_per_s.mean"),
    ("ram_gb_peak",           "resources.ram_gb.peak"),
    ("gpu_mem_gb_peak",       "resources.gpu_mem_gb.peak"),
]


@dataclass
class AggregateOptions:
    """Options for :func:`aggregate_dispatch`.

    Attributes:
        divergence_z: Z-score threshold for :attr:`divergent_seeds`; a seed
            is flagged when ``|value - mean| > divergence_z * std``.
            Default ``2.0``.
    """

    divergence_z: float = 2.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_dotted(d: dict, path: str) -> Any:
    for key in path.split("."):
        d = d[key]
    return d


def _load_bundle_jsons(bundle_dir: Path) -> tuple[dict, dict] | None:
    """Return ``(manifest, training)`` if both parse; ``None`` on any error."""
    try:
        with (bundle_dir / "manifest.json").open("r") as fh:
            manifest = json.load(fh)
        with (bundle_dir / "training.json").open("r") as fh:
            training = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return manifest, training


def _bundle_is_completed(manifest: dict, training: dict) -> bool:
    try:
        tr = manifest["phases"]["training"]
        if tr.get("status") != "completed" or tr.get("exit_code") != 0:
            return False
        ver = str(training.get("schema_version", ""))
        return ver.startswith("1.")
    except (KeyError, TypeError):
        return False


def _extract_seed_payload(training: dict, manifest: dict, run_id: str, assigned_to: str | None) -> dict:
    """Build one ``seeds[<N>]`` sub-object from a completed bundle."""
    rt = training["runtime"]
    rew = training["learning"]["reward"]
    ep = training["learning"]["ep_length"]
    res = training["resources"]
    startup_s = rt.get("startup_phase_times_s", {}) or {}
    return {
        "run_id":                  run_id,
        "status":                  "completed",
        "assigned_to":             assigned_to,
        "reward_final_ema":        float(rew["final_ema"]),
        "ep_length_final_ema":     float(ep["final_ema"]),
        "iter_time_s_mean":        float(rt["iteration_time_s"]["mean"]),
        "iter_time_s_std":         float(rt["iteration_time_s"]["std"]),
        "env_steps_per_s_mean":    float(rt["env_steps_per_s"]["mean"]),
        "iterations_completed":    int(rt["iterations_completed"]),
        "total_wall_time_s":       float(rt["total_wall_time_s"]),
        "ram_gb_peak":             float(res["ram_gb"]["peak"]),
        "gpu_mem_gb_peak":         float(res["gpu_mem_gb"]["peak"]),
        "startup_app_launch_s":    float(startup_s.get("app_launch") or 0.0),
        "startup_env_creation_s":  float(startup_s.get("env_creation") or 0.0),
        "startup_first_step_s":    float(startup_s.get("first_step") or 0.0),
    }


def _compute_aggregate(seeds: dict[str, dict], divergence_z: float) -> dict:
    """Compute the ``aggregate`` block + ``divergent_seeds`` for one row."""
    if not seeds:
        return {"n_seeds_completed": 0, "n_seeds_failed": 0}
    result: dict = {"n_seeds_completed": len(seeds), "n_seeds_failed": 0}
    for field_name, _ in _AGGREGATED_METRICS:
        values = [s[field_name] for s in seeds.values()]
        s = stats_over(values)
        result[field_name] = {
            "mean":   s.mean,
            "std":    s.std,
            "min":    s.min,
            "max":    s.max,
            "cv_pct": s.cv_pct,
        }
    return result


def _divergent_seed_keys(seeds: dict[str, dict], divergence_z: float) -> list[str]:
    """Return sorted seed-string keys flagged on ``reward_final_ema``."""
    if len(seeds) < 3:
        return []
    ordered = sorted(seeds.keys(), key=int)
    values = [seeds[k]["reward_final_ema"] for k in ordered]
    flagged_idx = is_divergent(values, z=divergence_z)
    return [ordered[i] for i in flagged_idx]


def _classify_failure(
    job: dict,
    bundle_dir: Path,
    bundle_jsons: tuple[dict, dict] | None,
) -> str:
    """Return the ``failure_kind`` string for a non-completed seed."""
    job_status = job.get("status")
    job_failure = job.get("failure") or {}
    job_failure_kind = job_failure.get("kind")

    if job_status == "failed" and job_failure_kind:
        return str(job_failure_kind)
    if not bundle_dir.exists():
        return "missing_bundle"
    if bundle_jsons is None:
        return "malformed_bundle"
    manifest, training = bundle_jsons
    if not _bundle_is_completed(manifest, training):
        return "malformed_bundle"
    # Shouldn't reach here — this job should have been classified completed.
    return "malformed_bundle"


def _classify_failure_message(job: dict) -> str:
    failure = job.get("failure") or {}
    return str(failure.get("message", ""))


def aggregate_dispatch(
    dispatch_dir: Path,
    options: AggregateOptions | None = None,
) -> dict:
    """Read ``dispatch_dir/dispatch.json`` + every ``<run_id>/`` bundle, return aggregate dict.

    Args:
        dispatch_dir: Path to ``odin_runs/<dispatch_id>/``.
        options: Optional :class:`AggregateOptions`; defaults to
            ``AggregateOptions()``.

    Returns:
        Aggregate dict matching T4.1 schema v1.0.

    Raises:
        FileNotFoundError: When ``dispatch_dir/dispatch.json`` is absent.
    """
    options = options or AggregateOptions()
    dispatch_path = dispatch_dir / "dispatch.json"
    if not dispatch_path.exists():
        raise FileNotFoundError(f"{dispatch_path} does not exist")
    with dispatch_path.open("r") as fh:
        dispatch = json.load(fh)

    rows_by_key: dict[tuple[str, str, str], dict] = {}
    failures: list[dict] = []
    commit_shas: list[str] = []
    hostnames: set[str] = set()

    for job in dispatch.get("jobs", []):
        run_id = job["run_id"]
        task = job["task_id"]
        framework = job["framework"]
        backend = job["backend"]
        seed = int(job["seed"])
        assigned_to = job.get("assigned_to")
        key = (task, framework, backend)
        if key not in rows_by_key:
            rows_by_key[key] = {
                "task":      task,
                "framework": framework,
                "backend":   backend,
                "seeds":     {},
            }

        bundle_dir = dispatch_dir / run_id
        bundle_jsons = _load_bundle_jsons(bundle_dir)
        is_completed = (
            bundle_dir.exists()
            and bundle_jsons is not None
            and _bundle_is_completed(*bundle_jsons)
        )
        if is_completed:
            manifest, training = bundle_jsons  # type: ignore[misc]
            rows_by_key[key]["seeds"][str(seed)] = _extract_seed_payload(
                training, manifest, run_id, assigned_to
            )
            commit_shas.append(str(manifest.get("machine", {}).get("git_commit") or ""))
            hostname = manifest.get("machine", {}).get("hostname")
            if hostname:
                hostnames.add(str(hostname))
        else:
            failures.append(
                {
                    "run_id":          run_id,
                    "task":            task,
                    "framework":       framework,
                    "backend":         backend,
                    "seed":            seed,
                    "assigned_to":     assigned_to,
                    "failure_kind":    _classify_failure(job, bundle_dir, bundle_jsons),
                    "failure_message": _classify_failure_message(job),
                }
            )

    rows: list[dict] = []
    for key in sorted(rows_by_key):
        r = rows_by_key[key]
        n_failed_for_row = sum(
            1
            for f in failures
            if (f["task"], f["framework"], f["backend"]) == key
        )
        agg = _compute_aggregate(r["seeds"], options.divergence_z)
        agg["n_seeds_failed"] = n_failed_for_row
        r["aggregate"] = agg if r["seeds"] else None
        r["divergent_seeds"] = _divergent_seed_keys(r["seeds"], options.divergence_z)
        rows.append(r)

    commit_sha = ""
    if commit_shas:
        counter = Counter(s for s in commit_shas if s)
        if counter:
            commit_sha, _ = counter.most_common(1)[0]
            minority = {s for s in counter if s != commit_sha}
            if minority:
                print(
                    f"[WARNING] bundles carried mixed commit SHAs; majority={commit_sha!r}, "
                    f"minority={sorted(minority)!r}"
                )

    totals = {
        "tasks":     len(rows),
        "runs":      sum(len(r["seeds"]) for r in rows) + len(failures),
        "completed": sum(len(r["seeds"]) for r in rows),
        "failed":    len(failures),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "dispatch_id":    str(dispatch.get("dispatch_id", dispatch_dir.name)),
        "generated_at":   _utc_now_iso(),
        "commit_sha":     commit_sha,
        "hostnames":      sorted(hostnames),
        "totals":         totals,
        "rows":           rows,
        "failures":       failures,
    }
```

- [ ] **Step 4: Run tests to verify happy path passes**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_aggregator.py::test_happy_path_two_completed_seeds -v --confcutdir=tools/odin
```

Expected: 1 PASS.

- [ ] **Step 5: Append the remaining test cases**

Append to `tools/odin/tests/test_valhalla_aggregator.py`:

```python
def test_divergent_seed_flagged(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run43", reward_final_ema=102.0)
    _write_completed_bundle(dispatch, "run44", reward_final_ema=101.0)
    _write_completed_bundle(dispatch, "run45", reward_final_ema=500.0)  # outlier
    _make_dispatch_json(
        dispatch,
        [_job(f"run{s}", seed=s) for s in (42, 43, 44, 45)],
    )
    agg = aggregate_dispatch(dispatch)
    row = agg["rows"][0]
    assert "45" in row["divergent_seeds"]
    assert "42" not in row["divergent_seeds"]


def test_mixed_completed_and_failed_seeds(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run43", reward_final_ema=110.0)
    # run44's bundle is missing entirely; job is marked failed in dispatch.json.
    _make_dispatch_json(
        dispatch,
        [
            _job("run42", seed=42),
            _job("run43", seed=43),
            _job(
                "run44",
                seed=44,
                status="failed",
                failure={"kind": "hugin_crash", "message": "RSL-RL subprocess exited 1", "details": {}},
            ),
        ],
    )
    agg = aggregate_dispatch(dispatch)
    row = agg["rows"][0]
    assert row["aggregate"]["n_seeds_completed"] == 2
    assert row["aggregate"]["n_seeds_failed"] == 1
    assert len(agg["failures"]) == 1
    f = agg["failures"][0]
    assert f["seed"] == 44
    assert f["failure_kind"] == "hugin_crash"
    assert f["failure_message"] == "RSL-RL subprocess exited 1"


def test_all_seeds_failed_row_has_null_aggregate(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _make_dispatch_json(
        dispatch,
        [
            _job(
                f"run{s}",
                seed=s,
                status="failed",
                failure={"kind": "hugin_crash", "message": "boom", "details": {}},
            )
            for s in (42, 43)
        ],
    )
    agg = aggregate_dispatch(dispatch)
    assert len(agg["rows"]) == 1
    row = agg["rows"][0]
    assert row["seeds"] == {}
    assert row["aggregate"] is None
    assert row["divergent_seeds"] == []
    assert len(agg["failures"]) == 2


def test_missing_bundle_synthesizes_missing_bundle_kind(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    # Job marked completed in dispatch.json but no bundle dir.
    _make_dispatch_json(
        dispatch,
        [_job("run42", seed=42, status="completed")],
    )
    agg = aggregate_dispatch(dispatch)
    assert len(agg["failures"]) == 1
    assert agg["failures"][0]["failure_kind"] == "missing_bundle"
    assert len(agg["rows"]) == 1
    assert agg["rows"][0]["seeds"] == {}


def test_malformed_training_json_wrong_schema_version(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    bundle = _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    # Corrupt training.json to schema v2.0 (major bump, rejected).
    with (bundle / "training.json").open("r") as fh:
        t = json.load(fh)
    t["schema_version"] = "2.0"
    with (bundle / "training.json").open("w") as fh:
        json.dump(t, fh)
    _make_dispatch_json(
        dispatch,
        [_job("run42", seed=42, status="completed")],
    )
    agg = aggregate_dispatch(dispatch)
    assert len(agg["failures"]) == 1
    assert agg["failures"][0]["failure_kind"] == "malformed_bundle"


def test_commit_sha_mismatch_majority_wins(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0, commit_sha="aaa111")
    _write_completed_bundle(dispatch, "run43", reward_final_ema=101.0, commit_sha="aaa111")
    _write_completed_bundle(dispatch, "run44", reward_final_ema=102.0, commit_sha="bbb222")
    _make_dispatch_json(
        dispatch,
        [_job(f"run{s}", seed=s) for s in (42, 43, 44)],
    )
    agg = aggregate_dispatch(dispatch)
    assert agg["commit_sha"] == "aaa111"
    captured = capsys.readouterr()
    assert "mixed commit SHAs" in captured.out
    assert "bbb222" in captured.out


def test_empty_dispatch_has_no_rows(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _make_dispatch_json(dispatch, [])
    agg = aggregate_dispatch(dispatch)
    assert agg["rows"] == []
    assert agg["failures"] == []
    assert agg["totals"]["tasks"] == 0
    assert agg["totals"]["runs"] == 0


def test_missing_dispatch_json_raises(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    with pytest.raises(FileNotFoundError):
        aggregate_dispatch(dispatch)
```

- [ ] **Step 6: Run all aggregator tests**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_aggregator.py -v --confcutdir=tools/odin
```

Expected: 8 PASS.

- [ ] **Step 7: Run pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/aggregator.py tools/odin/tests/test_valhalla_aggregator.py
git commit -m "Add Valhalla aggregator for per-dispatch rollups"
```

Subject is 48 chars — under 50.

---

### Task 3: `writer.py` — atomic JSON write with overwrite guard

**Files:**
- Create: `tools/odin/valhalla/writer.py`
- Create: `tools/odin/tests/test_valhalla_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/odin/tests/test_valhalla_writer.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.writer — atomic write of aggregate.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.writer import write_aggregate


def test_write_creates_file_with_expected_content(tmp_path: Path):
    agg = {"schema_version": "1.0", "dispatch_id": "20260423-100000", "rows": []}
    path = write_aggregate(tmp_path, agg)
    assert path == tmp_path / "aggregate.json"
    assert path.exists()
    with path.open("r") as fh:
        loaded = json.load(fh)
    assert loaded == agg


def test_write_overwrites_by_default(tmp_path: Path):
    (tmp_path / "aggregate.json").write_text('{"stale": true}')
    agg = {"schema_version": "1.0", "rows": [], "new": True}
    write_aggregate(tmp_path, agg)
    with (tmp_path / "aggregate.json").open("r") as fh:
        loaded = json.load(fh)
    assert loaded == agg


def test_write_no_overwrite_raises_on_existing(tmp_path: Path):
    (tmp_path / "aggregate.json").write_text('{"existing": true}')
    agg = {"schema_version": "1.0", "rows": []}
    with pytest.raises(FileExistsError):
        write_aggregate(tmp_path, agg, overwrite=False)


def test_write_cleans_up_temp_on_dump_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Sabotage json.dump so the writer has to clean up the tempfile.
    import tools.odin.valhalla.writer as mod

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated dump failure")

    monkeypatch.setattr(mod.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="simulated"):
        write_aggregate(tmp_path, {"x": 1})

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".aggregate_")]
    assert leftover == [], f"tempfile leaked: {leftover}"
    assert not (tmp_path / "aggregate.json").exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_writer.py -v --confcutdir=tools/odin
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `writer.py`**

Create `tools/odin/valhalla/writer.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic writer for ``<dispatch_dir>/aggregate.json``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = ["write_aggregate"]

_FILENAME = "aggregate.json"


def write_aggregate(
    dispatch_dir: Path,
    aggregate: dict,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write ``<dispatch_dir>/aggregate.json`` and return its path.

    Writes to a sibling temporary file then ``os.replace``\\ s over the
    final path, so a concurrent reader never observes a truncated file.

    Args:
        dispatch_dir: Target ``odin_runs/<dispatch_id>/`` directory.
        aggregate: Aggregate dict to serialize (matches T4.1 schema v1.0).
        overwrite: When ``False``, raises :class:`FileExistsError` if
            ``aggregate.json`` already exists. Default ``True``.

    Returns:
        Path to the written file.

    Raises:
        FileExistsError: When ``overwrite=False`` and the target exists.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    final_path = dispatch_dir / _FILENAME
    if not overwrite and final_path.exists():
        raise FileExistsError(f"{final_path} already exists (pass overwrite=True to replace)")

    fd, tmp_path_str = tempfile.mkstemp(prefix=".aggregate_", suffix=".json.tmp", dir=str(dispatch_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(aggregate, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return final_path
```

- [ ] **Step 4: Run tests + commit**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_writer.py -v --confcutdir=tools/odin
./isaaclab.sh -f
git add tools/odin/valhalla/writer.py tools/odin/tests/test_valhalla_writer.py
git commit -m "Add Valhalla atomic writer for aggregate.json"
```

Expected test output: 4 PASS. Subject 43 chars — under 50.

---

### Task 4: `cli.py` — `odin-aggregate <dispatch_id|LATEST>` entry

**Files:**
- Create: `tools/odin/valhalla/cli.py`
- Create: `tools/odin/tests/test_valhalla_cli.py`
- Create: `tools/odin/valhalla/__init__.py` (public re-exports)

- [ ] **Step 1: Write the CLI test**

Create `tools/odin/tests/test_valhalla_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.cli import main, parse_args, resolve_dispatch_dir


def _mkdir_dispatch(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir()
    return d


def test_parse_args_minimal():
    args = parse_args(["20260423-100000"])
    assert args.dispatch_id == "20260423-100000"
    assert str(args.runs_root) == "odin_runs"
    assert args.divergence_z == 2.0
    assert args.overwrite is True
    assert args.quiet is False


def test_parse_args_all_flags():
    args = parse_args(
        [
            "LATEST",
            "--runs-root",
            "/tmp/runs",
            "--divergence-z",
            "3.5",
            "--no-overwrite",
            "--quiet",
        ]
    )
    assert args.dispatch_id == "LATEST"
    assert str(args.runs_root) == "/tmp/runs"
    assert args.divergence_z == 3.5
    assert args.overwrite is False
    assert args.quiet is True


def test_resolve_dispatch_dir_exact_name(tmp_path: Path):
    d = _mkdir_dispatch(tmp_path, "20260423-100000")
    assert resolve_dispatch_dir(tmp_path, "20260423-100000") == d


def test_resolve_dispatch_dir_latest(tmp_path: Path):
    _mkdir_dispatch(tmp_path, "20260422-120000")
    newest = _mkdir_dispatch(tmp_path, "20260423-150000")
    _mkdir_dispatch(tmp_path, "20260423-100000")
    assert resolve_dispatch_dir(tmp_path, "LATEST") == newest


def test_resolve_dispatch_dir_latest_empty_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No prior dispatch"):
        resolve_dispatch_dir(tmp_path, "LATEST")


def test_resolve_dispatch_dir_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_dispatch_dir(tmp_path, "does-not-exist")


def test_cli_main_writes_aggregate(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dispatch = _mkdir_dispatch(tmp_path, "20260423-100000")
    (dispatch / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dispatch_id": "20260423-100000",
                "started_at": "2026-04-23T09:59:00Z",
                "ended_at": "2026-04-23T10:10:00Z",
                "seeds": [42],
                "commit_sha": "",
                "fleet": [],
                "jobs": [],
            }
        )
    )
    exit_code = main(["20260423-100000", "--runs-root", str(tmp_path)])
    assert exit_code == 0
    assert (dispatch / "aggregate.json").exists()
    out = capsys.readouterr().out
    assert "aggregate.json" in out or "rows" in out  # summary line printed


def test_cli_main_quiet_suppresses_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dispatch = _mkdir_dispatch(tmp_path, "20260423-100000")
    (dispatch / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dispatch_id": "20260423-100000",
                "started_at": "",
                "ended_at": "",
                "seeds": [42],
                "commit_sha": "",
                "fleet": [],
                "jobs": [],
            }
        )
    )
    exit_code = main(["20260423-100000", "--runs-root", str(tmp_path), "--quiet"])
    assert exit_code == 0
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_cli.py -v --confcutdir=tools/odin
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `cli.py`**

Create `tools/odin/valhalla/cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-aggregate — manual entry point for Valhalla aggregation.

Usage::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/cli.py <dispatch_id|LATEST> \\
        [--runs-root odin_runs/] \\
        [--divergence-z 2.0] \\
        [--no-overwrite] \\
        [--quiet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.odin.valhalla.aggregator import AggregateOptions, aggregate_dispatch
from tools.odin.valhalla.writer import write_aggregate


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI args. Factored out so unit tests can exercise just the argparse layer."""
    parser = argparse.ArgumentParser(
        prog="odin-aggregate",
        description="Roll an odin_runs/<dispatch_id>/ directory into a single aggregate.json.",
    )
    parser.add_argument(
        "dispatch_id",
        help="Dispatch id (matches odin_runs/<id>/), or LATEST to auto-pick the newest.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("odin_runs"))
    parser.add_argument("--divergence-z", type=float, default=2.0)
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Refuse to overwrite an existing aggregate.json (default: overwrite).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary line.")
    return parser.parse_args(argv)


def resolve_dispatch_dir(runs_root: Path, dispatch_id: str) -> Path:
    """Resolve ``dispatch_id`` (or ``LATEST``) to a concrete directory under ``runs_root``.

    Raises:
        FileNotFoundError: If ``LATEST`` is used but ``runs_root`` holds no
            subdirectories, or if ``dispatch_id`` names a non-existent directory.
    """
    if dispatch_id == "LATEST":
        children = sorted(p for p in runs_root.iterdir() if p.is_dir())
        if not children:
            raise FileNotFoundError(f"No prior dispatch directories under {runs_root}")
        return children[-1]
    candidate = runs_root / dispatch_id
    if not candidate.exists():
        raise FileNotFoundError(f"{candidate} does not exist")
    return candidate


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, non-zero on argparse or resolution errors.
    """
    args = parse_args(argv if argv is not None else [])
    dispatch_dir = resolve_dispatch_dir(args.runs_root, args.dispatch_id)
    agg = aggregate_dispatch(
        dispatch_dir,
        options=AggregateOptions(divergence_z=args.divergence_z),
    )
    path = write_aggregate(dispatch_dir, agg, overwrite=args.overwrite)
    if not args.quiet:
        totals = agg["totals"]
        print(
            f"Wrote {path}: {totals['tasks']} tasks, {totals['runs']} runs, "
            f"{totals['completed']} completed, {totals['failed']} failed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create `__init__.py` for public exports**

Create `tools/odin/valhalla/__init__.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Valhalla — per-dispatch aggregation of Odin bundles into aggregate.json."""

from tools.odin.valhalla.aggregator import AggregateOptions, aggregate_dispatch
from tools.odin.valhalla.stats import Stats, is_divergent, stats_over
from tools.odin.valhalla.writer import write_aggregate

__all__ = [
    "AggregateOptions",
    "Stats",
    "aggregate_dispatch",
    "is_divergent",
    "stats_over",
    "write_aggregate",
]
```

- [ ] **Step 5: Run tests + commit**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_cli.py -v --confcutdir=tools/odin
./isaaclab.sh -f
git add tools/odin/valhalla/cli.py tools/odin/valhalla/__init__.py tools/odin/tests/test_valhalla_cli.py
git commit -m "Add odin-aggregate CLI + valhalla package exports"
```

Expected test output: 7 PASS. Subject 48 chars — under 50.

---

### Task 5: Wire auto-aggregate into T3.1 `run_dispatch`

**Files:**
- Modify: `tools/odin/asgard/runner.py`
- Modify: `tools/odin/asgard/cli.py`
- Modify: `tools/odin/tests/test_asgard_runner.py` (add two tests)

- [ ] **Step 1: Read the current `run_dispatch` tail to locate the integration point**

```bash
grep -n "ended_at\|write_dispatch_state" tools/odin/asgard/runner.py | tail
```

Expected: the final block is:
```python
    state.ended_at = _utc_now_iso()
    write_dispatch_state(dispatch_dir, state)
    return state
```

- [ ] **Step 2: Write the failing tests**

Append to `tools/odin/tests/test_asgard_runner.py`:

```python
def test_run_dispatch_writes_aggregate_json(tmp_path, monkeypatch):
    """run_dispatch auto-invokes valhalla.aggregator at the tail."""
    from tools.odin.asgard.runner import DispatchOptions, run_dispatch

    fleet = _write_fleet(tmp_path)
    env_yaml = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "20260423-110000"
    dispatch_dir.mkdir()

    ssh = _FakeSSH()
    rsync = _FakeRsync()

    # Drain-loop completes immediately because _FakeWorker writes
    # completed state events — see existing test plumbing in this file.
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=env_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42]),
        ssh=ssh,
        rsync=rsync,
    )
    assert (dispatch_dir / "aggregate.json").exists()
    with (dispatch_dir / "aggregate.json").open("r") as fh:
        agg = __import__("json").load(fh)
    assert agg["schema_version"] == "1.0"


def test_run_dispatch_skip_aggregate_leaves_no_file(tmp_path, monkeypatch):
    """skip_aggregate=True suppresses the auto-call."""
    from tools.odin.asgard.runner import DispatchOptions, run_dispatch

    fleet = _write_fleet(tmp_path)
    env_yaml = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "20260423-110000"
    dispatch_dir.mkdir()

    ssh = _FakeSSH()
    rsync = _FakeRsync()

    run_dispatch(
        fleet=fleet,
        physx_yaml=env_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True),
        ssh=ssh,
        rsync=rsync,
    )
    assert not (dispatch_dir / "aggregate.json").exists()
```

**Note:** `_FakeSSH`, `_FakeRsync`, `_write_fleet`, `_write_env_list` are existing helpers in this file — reuse them unchanged. Check the top of the file if their signatures differ.

- [ ] **Step 3: Run tests to verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_runner.py::test_run_dispatch_writes_aggregate_json tools/odin/tests/test_asgard_runner.py::test_run_dispatch_skip_aggregate_leaves_no_file -v --confcutdir=tools/odin
```

Expected: FAIL — `skip_aggregate` is not a field of `DispatchOptions`; `aggregate.json` is not written.

- [ ] **Step 4: Add `skip_aggregate` to `DispatchOptions` and the auto-call**

In `tools/odin/asgard/runner.py`:

Find the `DispatchOptions` dataclass (around line 36) and add the new field:

```python
@dataclass
class DispatchOptions:
    """Options controlling one dispatch run.

    Args:
        seeds: RNG seeds to expand each env-list row across.
        max_infrastructure_retries: Per-job infrastructure retry cap passed to
            :class:`~tools.odin.asgard.worker.ValkyrieWorker`.
        per_job_timeout_s: Wall-clock timeout [s] per job.
        fresh: When ``True``, wipe and re-provision every host before running.
        skip_preflight: When ``True``, continue even if some hosts fail
            preflight (healthy-only dispatch).
        include_filter: Optional fnmatch patterns on ``task_id``; rows not
            matching any pattern are skipped.
        verbose: Print per-job completion lines to stdout.
        retry_failed: Explicit list of ``run_id`` values to re-attempt on a
            resume even though they are ``"failed"`` in the prior state.
        skip_aggregate: When ``True``, skip the automatic
            :func:`~tools.odin.valhalla.aggregate_dispatch` + write at the
            end of :func:`run_dispatch`. Default ``False``.
    """

    seeds: list[int]
    max_infrastructure_retries: int = 2
    per_job_timeout_s: int = 14400
    fresh: bool = False
    skip_preflight: bool = False
    include_filter: list[str] | None = None
    verbose: bool = False
    retry_failed: list[str] | None = None
    skip_aggregate: bool = False
```

Find the final block of `run_dispatch` (around line 400):

```python
    state.ended_at = _utc_now_iso()
    write_dispatch_state(dispatch_dir, state)
    return state
```

Replace with:

```python
    state.ended_at = _utc_now_iso()
    write_dispatch_state(dispatch_dir, state)

    if not options.skip_aggregate:
        try:
            from tools.odin.valhalla import aggregate_dispatch, write_aggregate

            agg = aggregate_dispatch(dispatch_dir)
            write_aggregate(dispatch_dir, agg)
        except Exception as exc:  # noqa: BLE001 — aggregate failure must not mask the dispatch return
            print(f"[WARNING] aggregate step failed: {exc}")

    return state
```

The `try/except` is deliberate: aggregation reads files the dispatch just wrote, but a schema-version mismatch or a malformed bundle shouldn't overwrite a successful dispatch's return value.

- [ ] **Step 5: Add `--skip-aggregate` passthrough to the Asgard CLI**

In `tools/odin/asgard/cli.py`, find the argparse section and add:

```python
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip the end-of-dispatch call to valhalla.aggregate_dispatch.",
    )
```

Then find where `DispatchOptions(...)` is constructed and add `skip_aggregate=args.skip_aggregate,` to the kwargs.

- [ ] **Step 6: Run tests**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_runner.py -v --confcutdir=tools/odin
```

Expected: all existing tests plus the 2 new tests pass.

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py tools/odin/asgard/cli.py tools/odin/tests/test_asgard_runner.py
git commit -m "Asgard: auto-aggregate at end of run_dispatch"
```

Subject 46 chars — under 50.

---

### Task 6: End-to-end integration test with T3.1

**Files:**
- Create: `tools/odin/tests/test_valhalla_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tools/odin/tests/test_valhalla_integration.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test: T3.1 run_dispatch + synthetic Hugin output → aggregate.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, run_dispatch
from tools.odin.asgard.transport import RsyncResult, SSHResult


# --- Fakes ---------------------------------------------------------------------------------------

@dataclass
class _FakeSSH:
    """SSH stub that writes synthetic bundles on `docker exec ... hugin` calls."""

    dispatch_dir: Path
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, env=None):
        del env, stdout_tee, timeout_s
        self.calls.append((host.host, list(cmd)))
        if "run.py" not in " ".join(cmd):
            return SSHResult(returncode=0, stdout="preflight ok", stderr="", duration_s=0.1)
        # Extract --run_id and materialize a synthetic bundle.
        try:
            run_id = cmd[cmd.index("--run_id") + 1]
        except (ValueError, IndexError):
            return SSHResult(returncode=1, stdout="", stderr="missing --run_id", duration_s=0.1)
        bundle = self.dispatch_dir / run_id
        bundle.mkdir(parents=True, exist_ok=True)
        _write_synthetic_bundle(bundle, run_id, seed=_extract_seed(run_id))
        return SSHResult(returncode=0, stdout="", stderr="", duration_s=0.1)


@dataclass
class _FakeRsync:
    def push(self, local, host, remote_path, *, exclude=(), delete=False):
        del local, host, remote_path, exclude, delete
        return RsyncResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    def pull(self, host, remote_path, local, *, exclude=(), delete=False):
        del host, remote_path, local, exclude, delete
        return RsyncResult(returncode=0, stdout="", stderr="", duration_s=0.1)


# --- Helpers -------------------------------------------------------------------------------------

def _extract_seed(run_id: str) -> int:
    # run_id format: rsl-rl_physx_<task>_<dispatch_id>_seed<N>
    return int(run_id.rsplit("seed", 1)[-1])


def _write_synthetic_bundle(bundle: Path, run_id: str, *, seed: int) -> None:
    # Reward varies by seed so aggregate stats are non-trivial.
    reward = 100.0 + seed * 0.5
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "machine": {"hostname": "valkyrie-01.internal", "git_commit": "abc123", "git_branch": "main"},
        "phases": {
            "startup": {"file": "startup.json", "status": "completed", "duration_s": 30.0, "exit_code": 0},
            "training": {"file": "training.json", "status": "completed", "duration_s": 150.0, "exit_code": 0},
        },
        "config": {
            "framework": "rsl_rl", "backend": "physx", "task": "Isaac-Ant-Direct-v0",
            "seed": seed, "num_envs": 4096, "max_iterations": 300,
        },
        "run_start_time_utc": "2026-04-23T10:00:00Z",
        "run_end_time_utc": "2026-04-23T10:03:00Z",
        "run_duration_s": 180.0,
        "artifacts": ["logs", "startup.json", "training.json", "training_data"],
    }
    training = {
        "schema_version": "1.0",
        "runtime": {
            "iterations_completed": 300,
            "total_wall_time_s": 150.0,
            "iteration_time_s": {"mean": 0.5, "std": 0.05},
            "env_steps_per_s": {"mean": 250000.0, "std": 2500.0},
            "iterations_per_s": {"mean": 2.0, "std": 0.01},
            "startup_phase_times_s": {"app_launch": 4.5, "env_creation": 12.4, "first_step": 0.006},
        },
        "resources": {
            "ram_gb": {"mean": 7.2, "peak": 8.0},
            "gpu_mem_gb": {"mean": 3.6, "peak": 4.0},
        },
        "learning": {
            "reward": {"final_raw": reward + 1, "final_ema": reward, "series_per_iter": [0.0] * 300},
            "ep_length": {"final_raw": 950, "final_ema": 940, "series_per_iter": [0.0] * 300},
        },
    }
    with (bundle / "manifest.json").open("w") as fh:
        json.dump(manifest, fh)
    with (bundle / "training.json").open("w") as fh:
        json.dump(training, fh)


def _make_env_yaml(tmp_path: Path) -> Path:
    from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list

    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            status="ok",
        )
    ]
    path = tmp_path / "physx.yaml"
    write_env_list(el, path)
    return path


# --- The test ------------------------------------------------------------------------------------

def test_run_dispatch_produces_aggregate_with_correct_stats(tmp_path: Path):
    fleet = Fleet(
        fleet_name="test",
        default_ssh_user="odin",
        default_ssh_key=None,
        hosts=[ValkyrieConfig(host="valkyrie-01.internal", ssh_user="odin", ssh_key=None, isaaclab_path="/opt/IsaacLab")],
    )
    physx_yaml = _make_env_yaml(tmp_path)
    dispatch_dir = tmp_path / "20260423-110000"
    dispatch_dir.mkdir()

    ssh = _FakeSSH(dispatch_dir=dispatch_dir)
    rsync = _FakeRsync()

    run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43, 44]),
        ssh=ssh,
        rsync=rsync,
    )

    agg_path = dispatch_dir / "aggregate.json"
    assert agg_path.exists()
    with agg_path.open("r") as fh:
        agg = json.load(fh)
    assert agg["schema_version"] == "1.0"
    assert agg["totals"]["tasks"] == 1
    assert agg["totals"]["completed"] == 3
    assert agg["totals"]["failed"] == 0
    row = agg["rows"][0]
    assert row["task"] == "Isaac-Ant-Direct-v0"
    assert row["aggregate"]["n_seeds_completed"] == 3
    # Seeds 42, 43, 44 → rewards 121.0, 121.5, 122.0 → mean 121.5
    assert row["aggregate"]["reward_final_ema"]["mean"] == 121.5
```

- [ ] **Step 2: Run the test**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_valhalla_integration.py -v --confcutdir=tools/odin
```

Expected: 1 PASS.

- [ ] **Step 3: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_valhalla_integration.py
git commit -m "Add Valhalla integration test with synthetic T3.1 dispatch"
```

Subject 57 chars — shorten: `Add Valhalla end-to-end integration test` (40).

---

### Task 7: Full-suite sweep + pre-commit

**Files:** none modified.

- [ ] **Step 1: Run the complete test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ -v --confcutdir=tools/odin
```

Expected: all 139 baseline tests + new Valhalla tests all pass. Rough expected count: ~158 passed + 1 skipped (the loopback SSH integration test).

- [ ] **Step 2: Pre-commit sweep**

```bash
./isaaclab.sh -f
```

Expected: all hooks pass. If any auto-fixes, stage and re-run.

- [ ] **Step 3: Commit pre-commit fixes if any**

```bash
# Only if there were modifications:
git add -u
git commit -m "Pre-commit auto-fixes on T4.1"
```

If pre-commit was already clean on step 2, skip this step.

- [ ] **Step 4: Update Odin architecture reference**

Edit `docs/odin/architecture.md`:
- Flip the T4 row in §6 task map from ⚪ to 🟡 (partial — T4.1 done, T4.2/T4.3 pending). Link the spec path:
  ```
  | T4 | Reporting + Valhalla dashboard (Layer 4) | `docs/superpowers/specs/2026-04-23-odin-t4-1-valhalla-aggregator-design.md` | 🟡 |
  ```
- Bump the "Last updated" line to `2026-04-23 (end of T4.1)`.
- Add a §9 change-log entry describing T4.1 deliverables:
  ```
  | 2026-04-23 | T4.1 delivered. `tools/odin/valhalla/` package (stats, aggregator, writer, cli) emits one `odin_runs/<dispatch_id>/aggregate.json` per dispatch: nested-per-row shape (`task × framework × backend` → per-seed drill-down + cross-seed `aggregate` block with mean/std/min/max/cv_pct on 6 headline metrics). `divergent_seeds` flags z-score outliers (default 2.0, tunable via CLI). Failed bundles (`missing_bundle`, `malformed_bundle`, or T3.1's four-way classification) land in top-level `failures[]`. Auto-generated at end of `run_dispatch` (skippable via `--skip-aggregate`); re-runnable via `odin-aggregate <dispatch_id|LATEST>`. 30+ unit tests + 1 integration test on synthetic dispatches. Real-fleet validation pass still pending — T4.1 is not "done" until one real dispatch on the runner machines has produced an `aggregate.json` the operator inspects. | Odin T4.1 |
  ```

- [ ] **Step 5: Commit doc update**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin T4.1 complete in architecture reference"
```

Subject 49 chars — under 50.

---

## Summary of verification criteria

After all 7 tasks, the following must hold:

- `./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ -v --confcutdir=tools/odin` → all pass.
- `./isaaclab.sh -f` → clean (no hook modifications).
- `tools/odin/valhalla/` package contains 5 modules + 4 new test files.
- T3.1 `run_dispatch` auto-produces `aggregate.json` unless `--skip-aggregate` is passed.
- `odin-aggregate LATEST` works from the repo root.
- The architecture doc flips T4 to 🟡 and carries a T4.1 change-log entry.

## What comes next (out of scope for this plan)

- **Real-fleet validation pass.** See spec §10. User plugs in runner machines, dispatches a curated subset, inspects the resulting aggregate.json, and we debug anything that surfaces. This is machine-time-bound and runs interactively after this plan lands.
- **T4.2 dashboard** (Dash/Plotly). Consumes the aggregate.json format this plan produces.
- **T4.3 baseline thresholds and regression indicator.** Future.
