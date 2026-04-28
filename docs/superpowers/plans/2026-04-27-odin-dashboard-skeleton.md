# Odin Dashboard Skeleton (Spec 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a usable Plotly Dash dashboard for Odin: `odin-dashboard` CLI, multi-dispatch landing page, header dispatch picker, three placeholder tabs, plus a small aggregator change that writes per-dispatch `hardware.json` for future hardware-fingerprint trend filtering.

**Architecture:** New sub-module `tools/odin/valhalla/dashboard/` with three responsibilities cleanly separated: `cli.py` (entry point, argparse, exit codes), `app.py` (Dash factory, routing, header), `data.py` (pure-Python data layer over `odin_runs/`). Aggregator gains a `_write_hardware_json` helper. Tabs use a registry pattern so Specs 1/2/3 plug in additively.

**Tech Stack:** Python 3.10+, Plotly Dash, pandas, plotly. No Isaac Sim runtime imports — all tests are pure-Python pytest.

**Branch:** `antoiner/feat/odin` (continues from spec commit `cf84ac3d330`).

**Spec:** `docs/superpowers/specs/2026-04-27-odin-dashboard-skeleton-design.md`.

---

## Conventions used in every task

- **Test runner:** `PYTHONPATH=. python3 -m pytest <test> -v --tb=short --noconftest -p no:cacheprovider`. The `--noconftest -p no:cacheprovider` flags bypass `tools/conftest.py` (which imports `isaaclab`) and keep runs hermetic. Pure-Python tests (`data.py`, aggregator hardware) don't need `dash` installed; tests that import `app.py` do.
- **Run tests one at a time** — never batch. Per the established discipline on this branch.
- **Pre-commit:** `./isaaclab.sh -f` BEFORE `git commit`. Restage modified files and rerun until clean.
- **Commit message:** Imperative subject ≤ 50 chars; body wrapped at 72 chars; **NO** AI co-authorship lines (per `AGENTS.md`).
- **No new system deps.** All deps are pip-installable Python packages.

## File map — what gets created or modified

| File | Owner task | Responsibility |
|---|---|---|
| `source/isaaclab/setup.py` | T1 | Add `dash`, `plotly`, `pandas` to `INSTALL_REQUIRES`. |
| `tools/odin/valhalla/dashboard/__init__.py` | T2 | Empty marker; exports `DataLayer`. |
| `tools/odin/valhalla/dashboard/data.py` | T2–T6 | `DispatchSummary` / `HardwareInfo` / `DataLayer` class. |
| `tools/odin/valhalla/dashboard/tests/__init__.py` | T2 | Empty marker. |
| `tools/odin/valhalla/dashboard/tests/test_data.py` | T2–T6 | DataLayer unit tests. |
| `tools/odin/valhalla/aggregator.py` | T7 | `_write_hardware_json` + integration into `aggregate_dispatch`. |
| `tools/odin/valhalla/dashboard/tests/test_aggregator_hardware.py` | T7 | Hardware-file writer tests. |
| `tools/odin/valhalla/dashboard/app.py` | T8 | `create_app(runs_root, initial_dispatch)` factory + routing. |
| `tools/odin/valhalla/dashboard/tabs/__init__.py` | T8 | Empty marker. |
| `tools/odin/valhalla/dashboard/tests/test_app.py` | T8 | App factory + routing tests. |
| `tools/odin/valhalla/dashboard/tests/test_app_landing.py` | T9 | Landing-table component tests. |
| `tools/odin/valhalla/dashboard/tabs/_placeholder.py` | T10 | "Tab not yet implemented" component. |
| `tools/odin/valhalla/dashboard/cli.py` | T11 | `odin-dashboard` entry point. |
| `tools/odin/valhalla/dashboard/tests/test_cli.py` | T11 | CLI argparse + exit code tests. |
| `tools/odin/valhalla/dashboard/assets/init.js` | T12 | One-line initial-dispatch redirect. |
| `tools/odin/tests/test_asgard_integration.py` | T13 | Existing slow-marked test gains hardware.json assertion. |
| `docs/odin/architecture.md` | T14 | Change-log entry. |

---

## Task 1: Add Dash dependencies

**Files:**
- Modify: `source/isaaclab/setup.py` (`INSTALL_REQUIRES`)

No tests for this — it's a one-line dep declaration. Verified by Task 8 (which imports `dash`). The implementer must also `pip install dash plotly pandas` in their test interpreter so subsequent tasks' tests can run.

- [ ] **Step 1.1: Locate `INSTALL_REQUIRES`**

Run: `grep -n "INSTALL_REQUIRES = \[" source/isaaclab/setup.py`
Expected: A line like `INSTALL_REQUIRES = [` near the top of the list.

- [ ] **Step 1.2: Add the three deps**

Add these three lines inside the `INSTALL_REQUIRES = [...]` block in `source/isaaclab/setup.py`. Place them just before the `# testing` comment (or anywhere natural — they're not testing-only):

```python
    # odin dashboard
    "dash>=2.18.0",
    "plotly>=5.24.0",
    "pandas>=2.0.0",
```

- [ ] **Step 1.3: Install the deps in the test interpreter**

Run (system Python; tests use this one):
```
pip install --user "dash>=2.18.0" "plotly>=5.24.0" "pandas>=2.0.0"
```

Expected: clean install. If they're already present, pip prints "Requirement already satisfied".

- [ ] **Step 1.4: Smoke-test imports**

Run: `python3 -c "import dash, plotly, pandas; print(dash.__version__, plotly.__version__, pandas.__version__)"`
Expected: three version strings printed; no `ImportError`.

- [ ] **Step 1.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/setup.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add dash / plotly / pandas to isaaclab install_requires

Hard deps for the Odin dashboard (Spec 0). dash provides the web
framework, plotly the chart library used inside dash components,
pandas the dataframe shape we'll feed the dispatch / aggregate
tables and the trend axis.
EOF
)"
```

---

## Task 2: `DataLayer` scaffold + `list_dispatches`

**Files:**
- Create: `tools/odin/valhalla/dashboard/__init__.py`
- Create: `tools/odin/valhalla/dashboard/data.py`
- Create: `tools/odin/valhalla/dashboard/tests/__init__.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_data.py`

This task delivers `DispatchSummary`, `HardwareInfo` dataclasses, the `DataLayer` class shell, and `list_dispatches`. Subsequent tasks add methods.

- [ ] **Step 2.1: Write the failing tests** — `tools/odin/valhalla/dashboard/tests/test_data.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.valhalla.dashboard.data`."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.dashboard.data import DataLayer, DispatchSummary


def _write_dispatch(runs_root: Path, dispatch_id: str, *, jobs: list[dict] | None = None,
                    started_at: str = "2026-04-27T14:13:02Z", ended_at: str | None = None) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "seeds": [42],
        "commit_sha": "abc123",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs or [],
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def test_list_dispatches_empty_runs_root(tmp_path):
    """Non-existent or empty runs_root returns []."""
    layer = DataLayer(tmp_path / "missing")
    assert layer.list_dispatches() == []

    (tmp_path / "empty").mkdir()
    layer = DataLayer(tmp_path / "empty")
    assert layer.list_dispatches() == []


def test_list_dispatches_returns_summary(tmp_path):
    """One dispatch with one completed job → one DispatchSummary."""
    _write_dispatch(
        tmp_path,
        "20260427-141302",
        ended_at="2026-04-27T14:30:00Z",
        jobs=[{"run_id": "r1", "status": "completed", "assigned_to": "v1"}],
    )
    layer = DataLayer(tmp_path)
    result = layer.list_dispatches()
    assert len(result) == 1
    s = result[0]
    assert isinstance(s, DispatchSummary)
    assert s.dispatch_id == "20260427-141302"
    assert s.started_at == "2026-04-27T14:13:02Z"
    assert s.ended_at == "2026-04-27T14:30:00Z"
    assert s.jobs_total == 1
    assert s.jobs_completed == 1
    assert s.jobs_failed == 0
    assert s.jobs_pending == 0
    assert s.skipped_total == 0
    assert s.hostnames == ["v1"]


def test_list_dispatches_excludes_loose_bundles(tmp_path):
    """Loose pre-T3.1 bundles (rsl-rl_..._seed42) are filtered out."""
    _write_dispatch(tmp_path, "20260427-141302")
    loose = tmp_path / "rsl-rl_physx_Isaac-Ant-Direct-v0_20260423-114242_seed42"
    loose.mkdir()
    (loose / "manifest.json").write_text("{}")
    layer = DataLayer(tmp_path)
    ids = [s.dispatch_id for s in layer.list_dispatches()]
    assert ids == ["20260427-141302"]


def test_list_dispatches_sorted_newest_first(tmp_path):
    """Three dispatches sorted descending by directory name."""
    _write_dispatch(tmp_path, "20260424-160119")
    _write_dispatch(tmp_path, "20260427-141302")
    _write_dispatch(tmp_path, "20260425-080000")
    layer = DataLayer(tmp_path)
    ids = [s.dispatch_id for s in layer.list_dispatches()]
    assert ids == ["20260427-141302", "20260425-080000", "20260424-160119"]


def test_list_dispatches_skips_dirs_without_dispatch_json(tmp_path):
    """A timestamp-named dir with no dispatch.json is skipped."""
    _write_dispatch(tmp_path, "20260427-141302")
    (tmp_path / "20260428-000000").mkdir()  # no dispatch.json
    layer = DataLayer(tmp_path)
    ids = [s.dispatch_id for s in layer.list_dispatches()]
    assert ids == ["20260427-141302"]


def test_list_dispatches_counts_skipped_jobs(tmp_path):
    """`skipped` array contributes to skipped_total."""
    d = _write_dispatch(tmp_path, "20260427-141302")
    payload = json.loads((d / "dispatch.json").read_text())
    payload["skipped"] = [
        {"task_id": "X", "framework": "rsl_rl", "backend": "physx", "seed": 42, "reason": "preset_unsupported"},
        {"task_id": "Y", "framework": "rsl_rl", "backend": "physx", "seed": 42, "reason": "native_backend_mismatch"},
    ]
    (d / "dispatch.json").write_text(json.dumps(payload))
    layer = DataLayer(tmp_path)
    s = layer.list_dispatches()[0]
    assert s.skipped_total == 2
```

- [ ] **Step 2.2: Run tests, verify they FAIL**

Run each test individually:

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_empty_runs_root -v --tb=short --noconftest -p no:cacheprovider
```

Expected: ALL fail with `ModuleNotFoundError: No module named 'tools.odin.valhalla.dashboard'`.

- [ ] **Step 2.3: Create `__init__.py` files (empty markers)**

Create `tools/odin/valhalla/dashboard/__init__.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin dashboard — Plotly Dash app over odin_runs/."""

from tools.odin.valhalla.dashboard.data import DataLayer, DispatchSummary, HardwareInfo

__all__ = ["DataLayer", "DispatchSummary", "HardwareInfo"]
```

Create `tools/odin/valhalla/dashboard/tests/__init__.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

- [ ] **Step 2.4: Implement `data.py`** with `DispatchSummary`, `HardwareInfo`, `DataLayer.__init__`, and `list_dispatches`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python data layer over ``odin_runs/`` for the Odin dashboard.

Zero Dash imports — exposed APIs are dataclasses and a single :class:`DataLayer`
class. Tab modules and the app shell call into this layer for everything that
touches disk so the UI layer stays orthogonal and the layer itself stays
trivially testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["DataLayer", "DispatchSummary", "HardwareInfo"]


_DISPATCH_ID_RE = re.compile(r"^\d{8}-\d{6}$")


@dataclass(frozen=True)
class DispatchSummary:
    """Headline view of one dispatch — what the landing table renders."""

    dispatch_id: str
    started_at: str
    ended_at: str | None
    jobs_total: int
    jobs_completed: int
    jobs_failed: int
    jobs_pending: int
    skipped_total: int
    hostnames: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HardwareInfo:
    """Per-host hardware block, normalized for cross-dispatch comparison."""

    hostname: str
    gpu_devices: list[dict[str, Any]]
    cpu_name: str
    cpu_count: int
    ram_gb: float
    sourced_from: str


class DataLayer:
    """All disk reads for the dashboard go through this class."""

    def __init__(self, runs_root: Path):
        self._runs_root = Path(runs_root).resolve() if runs_root else Path(runs_root)

    # -- list_dispatches ----------------------------------------------------

    def list_dispatches(self) -> list[DispatchSummary]:
        """Return all dispatches under ``runs_root``, newest-first.

        Filters to directories whose name matches ``YYYYMMDD-HHMMSS`` AND that
        contain a ``dispatch.json``. Loose pre-T3.1 bundles (e.g.
        ``rsl-rl_physx_..._seed42``) are excluded.
        """
        if not self._runs_root.exists():
            return []
        results: list[DispatchSummary] = []
        for entry in self._runs_root.iterdir():
            if not entry.is_dir():
                continue
            if not _DISPATCH_ID_RE.match(entry.name):
                continue
            dispatch_json = entry / "dispatch.json"
            if not dispatch_json.exists():
                continue
            try:
                payload = json.loads(dispatch_json.read_text())
            except json.JSONDecodeError:
                continue
            results.append(_summary_from_dispatch(payload))
        results.sort(key=lambda s: s.dispatch_id, reverse=True)
        return results


def _summary_from_dispatch(payload: dict[str, Any]) -> DispatchSummary:
    jobs = payload.get("jobs", []) or []
    by_status: dict[str, int] = {}
    for j in jobs:
        s = j.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    fleet = payload.get("fleet", []) or []
    hostnames = [h["host"] for h in fleet if "host" in h]
    return DispatchSummary(
        dispatch_id=str(payload.get("dispatch_id", "")),
        started_at=str(payload.get("started_at", "")),
        ended_at=payload.get("ended_at"),
        jobs_total=len(jobs),
        jobs_completed=by_status.get("completed", 0),
        jobs_failed=by_status.get("failed", 0),
        jobs_pending=by_status.get("pending", 0),
        skipped_total=len(payload.get("skipped", []) or []),
        hostnames=hostnames,
    )
```

- [ ] **Step 2.5: Run each test individually, verify all PASS**

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_empty_runs_root -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_returns_summary -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_excludes_loose_bundles -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_sorted_newest_first -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_skips_dirs_without_dispatch_json -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data.py::test_list_dispatches_counts_skipped_jobs -v --tb=short --noconftest -p no:cacheprovider
```

Expected: each → 1 passed in ~0.03s.

- [ ] **Step 2.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/__init__.py tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/__init__.py tools/odin/valhalla/dashboard/tests/test_data.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
DataLayer scaffold + list_dispatches

First slice of tools/odin/valhalla/dashboard/data.py: DispatchSummary
and HardwareInfo dataclasses, DataLayer class shell, list_dispatches
that filters timestamp-named dirs containing dispatch.json (loose
pre-T3.1 bundles excluded), sorted newest-first.
EOF
)"
```

---

## Task 3: `load_dispatch`, `load_aggregate`, `load_hardware`

**Files:**
- Modify: `tools/odin/valhalla/dashboard/data.py` (extend `DataLayer`)
- Modify: `tools/odin/valhalla/dashboard/tests/test_data.py`

- [ ] **Step 3.1: Append failing tests**

Append to `tools/odin/valhalla/dashboard/tests/test_data.py`:

```python
def test_load_dispatch_returns_payload(tmp_path):
    """load_dispatch returns the parsed dispatch.json dict."""
    _write_dispatch(tmp_path, "20260427-141302", ended_at="2026-04-27T14:30:00Z")
    layer = DataLayer(tmp_path)
    payload = layer.load_dispatch("20260427-141302")
    assert payload["dispatch_id"] == "20260427-141302"
    assert payload["ended_at"] == "2026-04-27T14:30:00Z"


def test_load_dispatch_raises_when_missing(tmp_path):
    """Unknown dispatch_id raises FileNotFoundError."""
    import pytest as _pytest

    layer = DataLayer(tmp_path)
    with _pytest.raises(FileNotFoundError):
        layer.load_dispatch("does-not-exist")


def test_load_aggregate_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    (d / "aggregate.json").write_text(json.dumps({"schema_version": "1.0", "rows": []}))
    layer = DataLayer(tmp_path)
    payload = layer.load_aggregate("20260427-141302")
    assert payload is not None
    assert payload["schema_version"] == "1.0"


def test_load_aggregate_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_aggregate("20260427-141302") is None


def test_load_hardware_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    (d / "hardware.json").write_text(json.dumps({
        "schema_version": "1.0",
        "dispatch_id": "20260427-141302",
        "fingerprint": "gpu:NVIDIA-L40",
        "hosts": {},
    }))
    layer = DataLayer(tmp_path)
    payload = layer.load_hardware("20260427-141302")
    assert payload is not None
    assert payload["fingerprint"] == "gpu:NVIDIA-L40"


def test_load_hardware_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_hardware("20260427-141302") is None
```

- [ ] **Step 3.2: Run new tests, verify they FAIL**

Run each individually with the same command shape. Expected: each → FAIL (`AttributeError: 'DataLayer' object has no attribute 'load_dispatch'` etc.).

- [ ] **Step 3.3: Add the three methods to `DataLayer`** in `tools/odin/valhalla/dashboard/data.py`

Insert these methods inside the `DataLayer` class, after `list_dispatches`:

```python
    # -- raw JSON readers ---------------------------------------------------

    def load_dispatch(self, dispatch_id: str) -> dict[str, Any]:
        """Read ``<runs_root>/<dispatch_id>/dispatch.json``.

        Raises:
            FileNotFoundError: if the file is absent.
        """
        path = self._runs_root / dispatch_id / "dispatch.json"
        if not path.exists():
            raise FileNotFoundError(f"dispatch.json missing for {dispatch_id} at {path}")
        return json.loads(path.read_text())

    def load_aggregate(self, dispatch_id: str) -> dict[str, Any] | None:
        """Read ``aggregate.json`` for the dispatch; ``None`` if absent."""
        path = self._runs_root / dispatch_id / "aggregate.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_hardware(self, dispatch_id: str) -> dict[str, Any] | None:
        """Read ``hardware.json`` for the dispatch; ``None`` if absent."""
        path = self._runs_root / dispatch_id / "hardware.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
```

- [ ] **Step 3.4: Run each new test individually, verify all PASS**

Same command shape. Expected: 6/6 pass.

- [ ] **Step 3.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
DataLayer: load_dispatch / load_aggregate / load_hardware

Three raw-JSON readers. load_dispatch raises FileNotFoundError on a
missing dispatch_id (programming error); load_aggregate and
load_hardware return None when their file is absent (legitimate state
during a live dispatch or a pre-feature run).
EOF
)"
```

---

## Task 4: `lookup_hardware` cross-dispatch fallback

**Files:**
- Modify: `tools/odin/valhalla/dashboard/data.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_data.py`

- [ ] **Step 4.1: Append failing tests**

```python
def _write_bundle(dispatch_dir: Path, run_id: str, *, hardware: dict | None = None) -> Path:
    bundle = dispatch_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    training = {"schema_version": "1.0", "hardware": hardware} if hardware else {"schema_version": "1.0"}
    (bundle / "training.json").write_text(json.dumps(training))
    return bundle


def test_lookup_hardware_returns_first_hit(tmp_path):
    """First newer dispatch wins when both have a bundle for the host."""
    older = _write_dispatch(
        tmp_path, "20260424-160119",
        jobs=[{"run_id": "old-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(older, "old-r1", hardware={
        "hostname": "Host-Old", "gpu_devices": [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
        "cpu_name": "Xeon", "cpu_count": 16, "ram_gb": 62.0,
    })
    newer = _write_dispatch(
        tmp_path, "20260427-141302",
        jobs=[{"run_id": "new-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(newer, "new-r1", hardware={
        "hostname": "Host-New", "gpu_devices": [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
        "cpu_name": "Xeon", "cpu_count": 16, "ram_gb": 62.79,
    })
    layer = DataLayer(tmp_path)
    info = layer.lookup_hardware("v1")
    assert info is not None
    assert info.hostname == "Host-New"
    assert info.sourced_from == "20260427-141302/new-r1"
    assert info.gpu_devices[0]["name"] == "NVIDIA L40"


def test_lookup_hardware_returns_none_when_unknown(tmp_path):
    """Host that never ran any bundle → None."""
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.lookup_hardware("never-seen") is None


def test_lookup_hardware_skips_bundles_without_hardware_block(tmp_path):
    """training.json without .hardware → skipped; falls through to next."""
    older = _write_dispatch(
        tmp_path, "20260424-160119",
        jobs=[{"run_id": "old-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(older, "old-r1", hardware={
        "hostname": "Host-Old", "gpu_devices": [],
        "cpu_name": "Xeon", "cpu_count": 8, "ram_gb": 32.0,
    })
    newer = _write_dispatch(
        tmp_path, "20260427-141302",
        jobs=[{"run_id": "new-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(newer, "new-r1", hardware=None)  # no .hardware block

    layer = DataLayer(tmp_path)
    info = layer.lookup_hardware("v1")
    assert info is not None
    # Newer bundle had no hardware → fell through to older.
    assert info.hostname == "Host-Old"
    assert info.sourced_from == "20260424-160119/old-r1"
```

- [ ] **Step 4.2: Run new tests, verify they FAIL**

Same command shape. Expected: each → FAIL.

- [ ] **Step 4.3: Add `lookup_hardware`** to `DataLayer`

Append after `load_hardware`:

```python
    # -- cross-dispatch lookup ----------------------------------------------

    def lookup_hardware(self, host: str) -> HardwareInfo | None:
        """Walk dispatches newest-first; return the first hardware block
        from any bundle whose ``assigned_to == host``.

        Used as a fall-back when a dispatch's own ``hardware.json`` is
        missing or doesn't list the host (e.g. for pre-feature dispatches).
        """
        for summary in self.list_dispatches():
            try:
                payload = self.load_dispatch(summary.dispatch_id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            jobs = payload.get("jobs", []) or []
            for job in jobs:
                if job.get("assigned_to") != host:
                    continue
                run_id = job.get("run_id")
                if not run_id:
                    continue
                training_path = self._runs_root / summary.dispatch_id / run_id / "training.json"
                if not training_path.exists():
                    continue
                try:
                    training = json.loads(training_path.read_text())
                except json.JSONDecodeError:
                    continue
                hw = training.get("hardware")
                if not hw:
                    continue
                return HardwareInfo(
                    hostname=str(hw.get("hostname", "")),
                    gpu_devices=list(hw.get("gpu_devices") or []),
                    cpu_name=str(hw.get("cpu_name", "")),
                    cpu_count=int(hw.get("cpu_count", 0)),
                    ram_gb=float(hw.get("ram_gb", 0.0)),
                    sourced_from=f"{summary.dispatch_id}/{run_id}",
                )
        return None
```

- [ ] **Step 4.4: Run each new test individually, verify all PASS**

- [ ] **Step 4.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
DataLayer: lookup_hardware cross-dispatch fall-back

Walks dispatches newest-first; returns the first training.json
.hardware block where assigned_to matches the requested host.
Used by Tab A (Spec 1) when the dispatch's own hardware.json is
absent or missing the host.
EOF
)"
```

---

## Task 5: `trend_dispatches_for` filter by hardware fingerprint

**Files:**
- Modify: `tools/odin/valhalla/dashboard/data.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_data.py`

- [ ] **Step 5.1: Append failing tests**

```python
def _write_aggregate_with_row(dispatch_dir: Path, *, task: str, framework: str, backend: str) -> None:
    payload = {
        "schema_version": "1.0",
        "rows": [{"task": task, "framework": framework, "backend": backend, "seeds": {}}],
    }
    (dispatch_dir / "aggregate.json").write_text(json.dumps(payload))


def _write_hardware(dispatch_dir: Path, fingerprint: str) -> None:
    payload = {
        "schema_version": "1.0",
        "dispatch_id": dispatch_dir.name,
        "fingerprint": fingerprint,
        "hosts": {},
    }
    (dispatch_dir / "hardware.json").write_text(json.dumps(payload))


def test_trend_filters_by_fingerprint(tmp_path):
    """Only dispatches with matching fingerprint appear in the trend."""
    a = _write_dispatch(tmp_path, "20260424-160119")
    _write_aggregate_with_row(a, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(a, "gpu:NVIDIA-L40")

    b = _write_dispatch(tmp_path, "20260425-080000")
    _write_aggregate_with_row(b, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(b, "gpu:NVIDIA-A100")

    c = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(c, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(c, "gpu:NVIDIA-L40")

    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=10)
    assert ids == ["20260427-141302", "20260424-160119"]


def test_trend_filters_by_task(tmp_path):
    """Dispatches without the requested (task, framework, backend) row are excluded."""
    a = _write_dispatch(tmp_path, "20260424-160119")
    _write_aggregate_with_row(a, task="Isaac-Cartpole-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(a, "gpu:NVIDIA-L40")

    b = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(b, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(b, "gpu:NVIDIA-L40")

    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=10)
    assert ids == ["20260427-141302"]


def test_trend_excludes_pre_feature_dispatches(tmp_path):
    """A dispatch without hardware.json is excluded from trends."""
    a = _write_dispatch(tmp_path, "20260424-160119")
    _write_aggregate_with_row(a, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    # NO hardware.json

    b = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(b, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(b, "gpu:NVIDIA-L40")

    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=10)
    assert ids == ["20260427-141302"]


def test_trend_returns_empty_when_current_has_no_hardware(tmp_path):
    """If the current dispatch has no hardware.json, trend is empty."""
    a = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(a, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx")
    assert ids == []


def test_trend_trims_to_n(tmp_path):
    """N=2 → only the two newest matches."""
    for did in ["20260423-000000", "20260424-000000", "20260425-000000", "20260426-000000"]:
        d = _write_dispatch(tmp_path, did)
        _write_aggregate_with_row(d, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
        _write_hardware(d, "gpu:NVIDIA-L40")
    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260426-000000", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=2)
    assert ids == ["20260426-000000", "20260425-000000"]
```

- [ ] **Step 5.2: Run new tests, verify they FAIL**

- [ ] **Step 5.3: Add `trend_dispatches_for`** to `DataLayer`

Append after `lookup_hardware`:

```python
    # -- trend axis ---------------------------------------------------------

    def trend_dispatches_for(
        self,
        current_dispatch_id: str,
        task: str,
        framework: str,
        backend: str,
        n: int = 10,
    ) -> list[str]:
        """Return the N most recent dispatch_ids that:

        - have a ``hardware.json`` whose fingerprint matches ``current_dispatch_id``
          (excludes pre-feature dispatches and mismatched-hardware dispatches), AND
        - have an ``aggregate.json`` row for ``(task, framework, backend)``.

        Sorted newest-first; trimmed to ``n``.
        """
        current_hw = self.load_hardware(current_dispatch_id)
        if current_hw is None:
            return []
        target_fingerprint = current_hw.get("fingerprint")
        if not target_fingerprint:
            return []
        matches: list[str] = []
        for summary in self.list_dispatches():
            hw = self.load_hardware(summary.dispatch_id)
            if hw is None or hw.get("fingerprint") != target_fingerprint:
                continue
            agg = self.load_aggregate(summary.dispatch_id)
            if agg is None:
                continue
            rows = agg.get("rows", []) or []
            if not any(
                r.get("task") == task and r.get("framework") == framework and r.get("backend") == backend
                for r in rows
            ):
                continue
            matches.append(summary.dispatch_id)
            if len(matches) >= n:
                break
        return matches
```

- [ ] **Step 5.4: Run each new test individually, verify all PASS**

- [ ] **Step 5.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
DataLayer: trend_dispatches_for hardware-fingerprint filter

Returns the N most-recent dispatches whose hardware.json fingerprint
matches the current dispatch and whose aggregate.json contains the
requested (task, framework, backend) row. Pre-feature dispatches
without hardware.json are excluded. Used by Tabs B and C (Specs 2/3)
to scope cross-commit trend axes to like-for-like comparisons.
EOF
)"
```

---

## Task 6: `load_training`, `load_startup`, `invalidate`

**Files:**
- Modify: `tools/odin/valhalla/dashboard/data.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_data.py`

- [ ] **Step 6.1: Append failing tests**

```python
def test_load_training_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    _write_bundle(d, "r1", hardware={
        "hostname": "h", "gpu_devices": [], "cpu_name": "x", "cpu_count": 1, "ram_gb": 1.0,
    })
    layer = DataLayer(tmp_path)
    payload = layer.load_training("20260427-141302", "r1")
    assert payload is not None
    assert payload["schema_version"] == "1.0"


def test_load_training_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_training("20260427-141302", "r-missing") is None


def test_load_startup_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    bundle = d / "r1"
    bundle.mkdir()
    (bundle / "startup.json").write_text(json.dumps({"schema_version": "1.0", "phases": {}}))
    layer = DataLayer(tmp_path)
    payload = layer.load_startup("20260427-141302", "r1")
    assert payload is not None
    assert payload["schema_version"] == "1.0"


def test_load_startup_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_startup("20260427-141302", "r-missing") is None


def test_invalidate_is_callable(tmp_path):
    """Smoke test: invalidate() with and without dispatch_id is a no-op
    on disk but must not raise. (Spec 1 will exercise the cache-drop
    behavior once methods are wrapped in lru_cache; for now it's a
    placeholder that lets callers invalidate state in advance.)"""
    layer = DataLayer(tmp_path)
    layer.invalidate()
    layer.invalidate("20260427-141302")
```

- [ ] **Step 6.2: Run new tests, verify they FAIL**

- [ ] **Step 6.3: Add three methods to `DataLayer`**

Append after `trend_dispatches_for`:

```python
    # -- per-bundle reads ---------------------------------------------------

    def load_training(self, dispatch_id: str, run_id: str) -> dict[str, Any] | None:
        """Read ``<runs_root>/<dispatch_id>/<run_id>/training.json``.

        Returns ``None`` when the file is absent (failed bundle, pulled-in-progress, etc.).
        """
        path = self._runs_root / dispatch_id / run_id / "training.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_startup(self, dispatch_id: str, run_id: str) -> dict[str, Any] | None:
        """Read ``<runs_root>/<dispatch_id>/<run_id>/startup.json``.

        Returns ``None`` when the file is absent.
        """
        path = self._runs_root / dispatch_id / run_id / "startup.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # -- cache control ------------------------------------------------------

    def invalidate(self, dispatch_id: str | None = None) -> None:
        """Drop cached state for ``dispatch_id`` (or all if ``None``).

        Callers (notably Tab A's poll on the live → done transition) call
        this before re-reading so the freshly-written aggregate.json /
        hardware.json is picked up. Spec 0 caches nothing yet — Specs 1+
        wrap reads in :func:`functools.lru_cache` and add cache-clear
        plumbing here. Defined now so callers don't need to be edited
        when caching arrives.
        """
        # Intentionally empty in Spec 0. See docstring.
        return None
```

- [ ] **Step 6.4: Run each test individually, verify all PASS**

- [ ] **Step 6.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
DataLayer: per-bundle readers + invalidate placeholder

load_training and load_startup return the parsed JSON or None. The
invalidate hook is wired in Spec 0 as a no-op so callers (Spec 1+)
can already call it; the actual cache-drop logic lands when
lru_cache wrappers are added.
EOF
)"
```

---

## Task 7: Aggregator writes `hardware.json`

**Files:**
- Modify: `tools/odin/valhalla/aggregator.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_aggregator_hardware.py`

- [ ] **Step 7.1: Read the existing aggregator** to find the right insertion point

Run: `grep -n "def aggregate_dispatch\|write_aggregate\|return result\|return aggregate" tools/odin/valhalla/aggregator.py`
Expected: locate the end of `aggregate_dispatch` where the function returns its result dict. The `hardware.json` write happens BEFORE the return, alongside other artifact writes.

Read the function's tail to confirm structure:
Run: `sed -n '160,260p' tools/odin/valhalla/aggregator.py`

- [ ] **Step 7.2: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_aggregator_hardware.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that ``aggregate_dispatch`` writes hardware.json."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.aggregator import aggregate_dispatch


def _write_dispatch_json(dispatch_dir: Path, jobs: list[dict]) -> None:
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_dir.name,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "seeds": [42],
        "commit_sha": "abc123",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs,
        "skipped": [],
    }
    (dispatch_dir / "dispatch.json").write_text(json.dumps(payload))


def _write_bundle(dispatch_dir: Path, run_id: str, hardware: dict | None) -> None:
    bundle = dispatch_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "framework": "rsl_rl",
        "backend": "physx",
        "task": "Isaac-Ant-Direct-v0",
        "seed": 42,
        "phases": {
            "training": {"file": "training.json", "status": "completed", "exit_code": 0, "duration_s": 10.0},
            "startup": {"file": "startup.json", "status": "completed", "exit_code": 0, "duration_s": 1.0},
        },
        "artifacts": [],
    }
    training: dict = {"schema_version": "1.0"}
    if hardware is not None:
        training["hardware"] = hardware
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "training.json").write_text(json.dumps(training))
    (bundle / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))


def test_aggregate_dispatch_writes_hardware_json(tmp_path):
    """A successful aggregate run produces hardware.json next to aggregate.json."""
    d = tmp_path / "20260427-141302"
    d.mkdir()
    _write_dispatch_json(d, [{
        "run_id": "rsl-rl_physx_Ant_seed42",
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": "rsl-rl_physx_Ant_seed42",
        "status": "completed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "preferred_not": [],
        "failure": None,
    }])
    _write_bundle(d, "rsl-rl_physx_Ant_seed42", hardware={
        "hostname": "Odin-Runner-5",
        "gpu_devices": [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
        "cpu_name": "Intel Xeon Processor (Icelake)",
        "cpu_count": 16,
        "ram_gb": 62.79,
    })

    aggregate_dispatch(d)

    hw_path = d / "hardware.json"
    assert hw_path.exists()
    payload = json.loads(hw_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["dispatch_id"] == "20260427-141302"
    assert "generated_at" in payload
    assert payload["fingerprint"] == "gpu:NVIDIA-L40"
    assert "v1" in payload["hosts"]
    block = payload["hosts"]["v1"]
    assert block["hostname"] == "Odin-Runner-5"
    assert block["gpu_devices"] == [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}]
    assert block["cpu_name"] == "Intel Xeon Processor (Icelake)"
    assert block["cpu_count"] == 16
    assert block["ram_gb"] == 62.79
    assert block["sourced_from"] == "rsl-rl_physx_Ant_seed42"


def test_aggregate_dispatch_skips_hardware_when_no_completed_bundles(tmp_path):
    """No completed bundles → no hardware.json (warning logged, aggregate still written)."""
    d = tmp_path / "20260427-141302"
    d.mkdir()
    _write_dispatch_json(d, [{
        "run_id": "rsl-rl_physx_Ant_seed42",
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": "rsl-rl_physx_Ant_seed42",
        "status": "failed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "preferred_not": [],
        "failure": {"kind": "hugin_crash", "message": "x", "details": {}},
    }])

    aggregate_dispatch(d)

    assert not (d / "hardware.json").exists()


def test_aggregate_dispatch_skips_hardware_when_training_lacks_block(tmp_path):
    """Bundle exists but training.json has no .hardware → no hardware.json."""
    d = tmp_path / "20260427-141302"
    d.mkdir()
    _write_dispatch_json(d, [{
        "run_id": "rsl-rl_physx_Ant_seed42",
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": "rsl-rl_physx_Ant_seed42",
        "status": "completed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "preferred_not": [],
        "failure": None,
    }])
    _write_bundle(d, "rsl-rl_physx_Ant_seed42", hardware=None)

    aggregate_dispatch(d)

    # Aggregator does not raise; hardware.json simply not written.
    assert not (d / "hardware.json").exists()
```

- [ ] **Step 7.3: Run new tests, verify they FAIL**

Same individual command shape per test. Expected: each → FAIL (`hardware.json` doesn't exist after `aggregate_dispatch`).

- [ ] **Step 7.4: Modify `tools/odin/valhalla/aggregator.py`**

Add this helper function before `aggregate_dispatch`:

```python
def _write_hardware_json(dispatch_dir: Path, dispatch_payload: dict) -> None:
    """Emit ``<dispatch_dir>/hardware.json`` for the dashboard's hardware-fingerprint trend filter.

    Reads ``training.json.hardware`` from the first completed bundle per host,
    builds a per-host map, and computes a single ``fingerprint`` from the
    first host's first GPU. Failures are logged and swallowed — aggregator
    must continue regardless.
    """
    import os
    import tempfile
    from datetime import datetime, timezone

    try:
        jobs = dispatch_payload.get("jobs", []) or []
        seen_hosts: dict[str, dict] = {}
        for job in jobs:
            if job.get("status") != "completed":
                continue
            host = job.get("assigned_to")
            run_id = job.get("run_id")
            if not host or not run_id or host in seen_hosts:
                continue
            training_path = dispatch_dir / run_id / "training.json"
            if not training_path.exists():
                continue
            try:
                training = json.loads(training_path.read_text())
            except json.JSONDecodeError:
                continue
            hw = training.get("hardware")
            if not hw:
                continue
            seen_hosts[host] = {
                "hostname": str(hw.get("hostname", "")),
                "gpu_devices": list(hw.get("gpu_devices") or []),
                "cpu_name": str(hw.get("cpu_name", "")),
                "cpu_count": int(hw.get("cpu_count", 0)),
                "ram_gb": float(hw.get("ram_gb", 0.0)),
                "sourced_from": run_id,
            }
        if not seen_hosts:
            print(f"[WARNING] hardware.json: no completed bundle with .hardware block in {dispatch_dir}")
            return
        first_host_block = next(iter(seen_hosts.values()))
        gpus = first_host_block["gpu_devices"]
        if not gpus:
            print(f"[WARNING] hardware.json: first host has no GPU devices in {dispatch_dir}")
            return
        gpu_name = str(gpus[0].get("name", "")).strip()
        if not gpu_name:
            print(f"[WARNING] hardware.json: GPU name empty in {dispatch_dir}")
            return
        fingerprint_name = gpu_name.replace(" ", "-")
        payload = {
            "schema_version": "1.0",
            "dispatch_id": dispatch_payload.get("dispatch_id", dispatch_dir.name),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hosts": seen_hosts,
            "fingerprint": f"gpu:{fingerprint_name}",
        }
        out = dispatch_dir / "hardware.json"
        # Atomic write: temp + rename.
        fd, tmp_path = tempfile.mkstemp(prefix="hardware.json.", dir=str(dispatch_dir))
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, out)
    except Exception as exc:  # noqa: BLE001 — best-effort, never block aggregate
        print(f"[WARNING] hardware.json: write failed in {dispatch_dir}: {exc}")
```

Then, inside `aggregate_dispatch`, after `dispatch = json.load(fh)` is read AND after the rows are computed (basically anywhere it's safe to call before the return), add the hardware-write call:

Locate the line near the end of `aggregate_dispatch` that returns the result dict (likely around `return result` or `return {...}`). Just BEFORE that return, insert:

```python
    _write_hardware_json(dispatch_dir, dispatch)
```

If the variable holding the parsed dispatch.json dict is named differently in your local copy, use that name. Verify by reading the function tail before the edit.

- [ ] **Step 7.5: Run each new test individually, verify all PASS**

- [ ] **Step 7.6: Run the full aggregator test file** to confirm no regression

```
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_valhalla_writer.py --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_valhalla_cli.py --tb=short --noconftest -p no:cacheprovider
```

Expected: all pre-existing tests still pass.

- [ ] **Step 7.7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/aggregator.py tools/odin/valhalla/dashboard/tests/test_aggregator_hardware.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Aggregator: write per-dispatch hardware.json

After aggregate.json is built, _write_hardware_json reads the first
completed bundle per host's training.json.hardware block, assembles a
per-host map keyed by host address, computes fingerprint as
'gpu:<gpu_name>' from the first host's first GPU, and writes
hardware.json next to aggregate.json. Failure is logged with a
[WARNING] line; never blocks the aggregate.

Used by the dashboard's trend axis (Specs 2/3) to scope cross-commit
comparisons to dispatches with matching hardware fingerprint.
EOF
)"
```

---

## Task 8: `app.py` factory + routing + 404

**Files:**
- Create: `tools/odin/valhalla/dashboard/app.py`
- Create: `tools/odin/valhalla/dashboard/tabs/__init__.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_app.py`

This task delivers `create_app(runs_root, initial_dispatch=None) -> dash.Dash` with the layout skeleton, URL routing, and 404 handling. **Tabs are placeholders** — Task 10 adds `tabs/_placeholder.py`.

- [ ] **Step 8.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_app.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Dash app factory + routing."""

from __future__ import annotations

import json
from pathlib import Path

import dash

from tools.odin.valhalla.dashboard.app import create_app, route_pathname


def _write_dispatch(runs_root: Path, dispatch_id: str) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "abc",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": [],
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def test_create_app_returns_dash_instance(tmp_path):
    app = create_app(tmp_path)
    assert isinstance(app, dash.Dash)
    assert app.title == "Odin"


def test_create_app_layout_is_non_empty(tmp_path):
    app = create_app(tmp_path)
    assert app.layout is not None


def test_route_pathname_landing(tmp_path):
    """Empty path returns the landing component."""
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/", data)
    # Identify by a stable id we set on the landing root component.
    assert _has_id(component, "landing-root")


def test_route_pathname_dispatch_redirects_to_tab_a(tmp_path):
    """`/<id>/` returns a redirect (Location component) to /<id>/dispatch-fleet."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/", data)
    assert _has_id(component, "redirect-to-tab-a") or _is_redirect_to(component, "/20260427-141302/dispatch-fleet")


def test_route_pathname_unknown_dispatch_returns_404(tmp_path):
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/does-not-exist/dispatch-fleet", data)
    assert _has_id(component, "not-found-root")


def test_route_pathname_unknown_path_returns_404(tmp_path):
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/garbage/route/here", data)
    assert _has_id(component, "not-found-root")


def test_route_pathname_known_tab_renders_placeholder(tmp_path):
    """Tab path on a real dispatch renders the placeholder (Spec 0 has no tab content)."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/dispatch-fleet", data)
    assert _has_id(component, "tab-placeholder")


# -- helpers --


def _walk(component):
    """Yield this component plus every descendant in its `children` tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None or isinstance(child, str):
                continue
            yield from _walk(child)
    else:
        if not isinstance(children, str):
            yield from _walk(children)


def _has_id(component, target_id: str) -> bool:
    for c in _walk(component):
        if getattr(c, "id", None) == target_id:
            return True
    return False


def _is_redirect_to(component, expected_href: str) -> bool:
    """A dcc.Location with the expected href is acceptable for redirect."""
    for c in _walk(component):
        if isinstance(c, dash.dcc.Location):
            if getattr(c, "href", None) == expected_href and getattr(c, "refresh", False):
                return True
    return False
```

- [ ] **Step 8.2: Run tests, verify they FAIL**

Run each individually with the same command shape. Expected: each → FAIL with `ModuleNotFoundError: No module named 'tools.odin.valhalla.dashboard.app'`.

- [ ] **Step 8.3: Create `tools/odin/valhalla/dashboard/tabs/__init__.py`** (empty)

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

- [ ] **Step 8.4: Implement `app.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plotly Dash app factory for the Odin dashboard.

Builds the SPA shell: header (logo + dispatch dropdown + live/done pill),
tab strip, page-content area. Routing lives in :func:`route_pathname` so it
can be unit-tested without the live Dash callback machinery.
"""

from __future__ import annotations

import re
from pathlib import Path

import dash
from dash import Input, Output, dcc, html

from tools.odin.valhalla.dashboard.data import DataLayer

__all__ = ["create_app", "route_pathname"]


_DISPATCH_ID_RE = re.compile(r"^\d{8}-\d{6}$")
_TAB_IDS = {"dispatch-fleet", "task-drilldown", "startup"}


def create_app(runs_root: Path, initial_dispatch: Path | None = None) -> dash.Dash:
    """Build the Dash app. Pure factory — no global state."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Odin"
    data = DataLayer(runs_root)
    app.layout = _build_layout(initial_dispatch)
    _register_callbacks(app, data)
    return app


def _build_layout(initial_dispatch: Path | None) -> html.Div:
    initial_path = "/"
    if initial_dispatch is not None:
        initial_path = f"/{initial_dispatch.name}/dispatch-fleet"
    return html.Div(
        id="app-root",
        children=[
            dcc.Location(id="url", refresh=False, pathname=initial_path),
            dcc.Store(id="active-dispatch", storage_type="memory"),
            html.Div(id="page-content"),
        ],
    )


def _register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def _on_url(pathname: str):
        return route_pathname(pathname or "/", data)


def route_pathname(pathname: str, data: DataLayer):
    """Map a URL pathname to the Dash component tree to render at /page-content.

    Pulled out as a free function so unit tests can drive routing without
    spinning up the Dash callback graph.
    """
    parts = [p for p in pathname.split("/") if p]
    if not parts:
        return _landing(data)
    dispatch_id = parts[0]
    if not _DISPATCH_ID_RE.match(dispatch_id):
        return _not_found(pathname)
    # Verify the dispatch actually exists.
    try:
        data.load_dispatch(dispatch_id)
    except FileNotFoundError:
        return _not_found(pathname)
    if len(parts) == 1:
        # /<id>/ → redirect to default tab
        return html.Div(
            id="redirect-to-tab-a",
            children=[
                dcc.Location(id="redirect-loc", href=f"/{dispatch_id}/dispatch-fleet", refresh=True),
            ],
        )
    tab_id = parts[1]
    if tab_id not in _TAB_IDS:
        return _not_found(pathname)
    return _render_tab(dispatch_id, tab_id, data)


def _landing(data: DataLayer) -> html.Div:
    """Multi-dispatch landing: real list of dispatches.

    Spec 0 ships a minimal stub here so the routing test passes; Task 9
    fills in the actual table.
    """
    return html.Div(id="landing-root", children=[html.H2("Odin dashboard"), html.Div(id="landing-table")])


def _not_found(pathname: str) -> html.Div:
    return html.Div(
        id="not-found-root",
        children=[
            html.H2("Not found"),
            html.P(f"No route for {pathname!r}."),
            dcc.Link("Back to dashboard", href="/"),
        ],
    )


def _render_tab(dispatch_id: str, tab_id: str, data: DataLayer) -> html.Div:
    """Render the tab body for /<id>/<tab_id>.

    Spec 0 returns the placeholder for every tab. Tab-specific specs (1/2/3)
    add their own modules under ``dashboard/tabs/`` that override this via
    a registry; but Spec 0 doesn't depend on that wiring being present.
    """
    from tools.odin.valhalla.dashboard.tabs import _placeholder

    return _placeholder.render(dispatch_id, tab_id)
```

- [ ] **Step 8.5: Implement a minimal `tabs/_placeholder.py`** so app tests can load it

The full placeholder is delivered in Task 10, but `app.py` imports it. Add this stub now:

`tools/odin/valhalla/dashboard/tabs/_placeholder.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spec-0 placeholder rendered for any tab whose real implementation hasn't landed yet."""

from __future__ import annotations

from dash import html


def render(dispatch_id: str, tab_id: str):
    """Return the Spec-0 placeholder component for ``<dispatch_id>/<tab_id>``."""
    spec_number = {"dispatch-fleet": 1, "task-drilldown": 2, "startup": 3}.get(tab_id, "?")
    return html.Div(
        id="tab-placeholder",
        children=[
            html.H3(f"Tab '{tab_id}'"),
            html.P(
                f"Coming in Spec {spec_number}. Dashboard skeleton (Spec 0) ships only "
                f"the multi-dispatch routing and dispatch picker.",
            ),
            html.P(f"Active dispatch: {dispatch_id}"),
        ],
    )
```

(Task 10 will visit this file again to confirm/polish; the contract here is the `render(dispatch_id, tab_id)` signature.)

- [ ] **Step 8.6: Run each test individually, verify all PASS**

- [ ] **Step 8.7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/app.py tools/odin/valhalla/dashboard/tabs/__init__.py tools/odin/valhalla/dashboard/tabs/_placeholder.py tools/odin/valhalla/dashboard/tests/test_app.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Dash app factory + routing skeleton

create_app(runs_root, initial_dispatch=None) returns a configured
Dash instance with the SPA shell (Location, Store, page-content
slot). Routing lives in route_pathname() as a pure function so it
can be unit-tested without the callback graph: /<id>/<tab> →
placeholder; /<id>/ → redirect to default tab; bad path → 404.
EOF
)"
```

---

## Task 9: Landing-page table

**Files:**
- Modify: `tools/odin/valhalla/dashboard/app.py` (`_landing`)
- Create: `tools/odin/valhalla/dashboard/tests/test_app_landing.py`

- [ ] **Step 9.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_app_landing.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-dispatch landing page."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.dashboard.app import _landing
from tools.odin.valhalla.dashboard.data import DataLayer


def _write_dispatch(runs_root: Path, dispatch_id: str, *, jobs_total: int = 0,
                    completed: int = 0, failed: int = 0, ended_at: str | None = None) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(completed):
        jobs.append({"run_id": f"c-{i}", "status": "completed", "assigned_to": "v1"})
    for i in range(failed):
        jobs.append({"run_id": f"f-{i}", "status": "failed", "assigned_to": "v1"})
    while len(jobs) < jobs_total:
        jobs.append({"run_id": f"p-{len(jobs)}", "status": "pending", "assigned_to": None})
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": ended_at,
        "seeds": [42],
        "commit_sha": "abc",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs,
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None or isinstance(child, str):
                continue
            yield from _walk(child)
    elif not isinstance(children, str):
        yield from _walk(children)


def test_landing_with_no_dispatches_renders_empty_state(tmp_path):
    """No dispatches → an empty-state message instead of an empty table."""
    component = _landing(DataLayer(tmp_path))
    text_components = [c for c in _walk(component) if hasattr(c, "children") and isinstance(getattr(c, "children", None), str)]
    text_blob = " ".join(c.children for c in text_components)
    assert "No dispatches" in text_blob


def test_landing_renders_one_row_per_dispatch(tmp_path):
    """Three dispatches → three table rows (in addition to the header)."""
    _write_dispatch(tmp_path, "20260427-141302", jobs_total=3, completed=2, failed=1, ended_at="2026-04-27T14:30:00Z")
    _write_dispatch(tmp_path, "20260425-080000", jobs_total=4, completed=4, ended_at="2026-04-25T08:30:00Z")
    _write_dispatch(tmp_path, "20260424-160119", jobs_total=15, completed=15, ended_at="2026-04-24T16:30:00Z")

    component = _landing(DataLayer(tmp_path))
    table_rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    # 1 header row + 3 data rows
    assert len(table_rows) == 4


def test_landing_row_contains_dispatch_id_and_link(tmp_path):
    """Each row has the dispatch_id text and a link to /<id>/."""
    _write_dispatch(tmp_path, "20260427-141302", jobs_total=1)
    component = _landing(DataLayer(tmp_path))
    links = [c for c in _walk(component) if type(c).__name__ == "A"]
    hrefs = [getattr(link, "href", None) for link in links]
    assert "/20260427-141302/" in hrefs


def test_landing_sorts_newest_first(tmp_path):
    """Three dispatches → rendered newest-first in the rendered DOM."""
    _write_dispatch(tmp_path, "20260424-160119", jobs_total=1)
    _write_dispatch(tmp_path, "20260427-141302", jobs_total=1)
    _write_dispatch(tmp_path, "20260425-080000", jobs_total=1)

    component = _landing(DataLayer(tmp_path))
    # Collect anchor hrefs in DOM order — they should match newest-first.
    links = [c for c in _walk(component) if type(c).__name__ == "A"]
    hrefs = [getattr(link, "href", "") for link in links]
    dispatch_hrefs = [h for h in hrefs if h.startswith("/2026")]
    assert dispatch_hrefs == [
        "/20260427-141302/",
        "/20260425-080000/",
        "/20260424-160119/",
    ]
```

- [ ] **Step 9.2: Run new tests, verify they FAIL**

Same individual command shape. Expected: each → FAIL because `_landing` currently returns the stub.

- [ ] **Step 9.3: Replace `_landing` in `app.py`** with the real table

Replace the existing `_landing` function with:

```python
def _landing(data: DataLayer) -> html.Div:
    """Multi-dispatch landing: real table of dispatches sorted newest-first."""
    summaries = data.list_dispatches()
    if not summaries:
        return html.Div(
            id="landing-root",
            children=[
                html.H2("Odin dashboard"),
                html.P("No dispatches under runs_root yet. Run odin-dispatch to create one."),
            ],
        )
    header = html.Tr(
        children=[
            html.Th("Dispatch"),
            html.Th("Started"),
            html.Th("Ended"),
            html.Th("Total"),
            html.Th("Completed"),
            html.Th("Failed"),
            html.Th("Pending"),
            html.Th("Skipped"),
            html.Th("Hosts"),
        ],
    )
    rows = [
        html.Tr(
            children=[
                html.Td(html.A(s.dispatch_id, href=f"/{s.dispatch_id}/")),
                html.Td(s.started_at or "—"),
                html.Td(s.ended_at or "—"),
                html.Td(str(s.jobs_total)),
                html.Td(str(s.jobs_completed)),
                html.Td(str(s.jobs_failed)),
                html.Td(str(s.jobs_pending)),
                html.Td(str(s.skipped_total)),
                html.Td(", ".join(s.hostnames) or "—"),
            ],
        )
        for s in summaries
    ]
    return html.Div(
        id="landing-root",
        children=[
            html.H2("Odin dashboard"),
            html.Table(children=[html.Thead(children=[header]), html.Tbody(children=rows)]),
        ],
    )
```

- [ ] **Step 9.4: Run each new test individually, verify all PASS**

- [ ] **Step 9.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/app.py tools/odin/valhalla/dashboard/tests/test_app_landing.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Dash landing page renders dispatch table

_landing builds a sortable table from data.list_dispatches() with
columns: dispatch_id link, started/ended timestamps, totals
(total/completed/failed/pending/skipped), hostnames. Empty fleet =
empty-state message. Spec 0 ships a working dashboard the moment it
lands — tabs are placeholders, but the landing page is real.
EOF
)"
```

---

## Task 10: Tab placeholder polish + registry helper

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/_placeholder.py`
- Modify: `tools/odin/valhalla/dashboard/app.py` (registry hook)
- Modify: `tools/odin/valhalla/dashboard/tests/test_app.py`

The placeholder was sketched in Task 8. Now we (a) wire a `_discover_tabs()` registry helper that imports any of `tabs/dispatch_fleet.py` / `tabs/task_drilldown.py` / `tabs/startup.py` if present (none in Spec 0), so Specs 1/2/3 are pure additive; (b) add a test for the placeholder content; (c) make `_render_tab` prefer real tab modules over the placeholder when they exist.

- [ ] **Step 10.1: Append failing test** to `tools/odin/valhalla/dashboard/tests/test_app.py`

```python
def test_placeholder_mentions_target_spec(tmp_path):
    """Placeholder text names which spec implements the tab."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer

    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/task-drilldown", data)
    blobs = [c.children for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(blobs)
    assert "Spec 2" in text


def test_real_tab_module_overrides_placeholder(tmp_path, monkeypatch):
    """If `tabs/dispatch_fleet.py` exists with render(), it's used in place of placeholder."""
    import sys
    import types

    fake_module = types.ModuleType("tools.odin.valhalla.dashboard.tabs.dispatch_fleet")

    def _render(dispatch_id, tab_id):
        from dash import html
        return html.Div(id="real-tab-a", children=[html.P(f"Real Tab A for {dispatch_id}")])

    fake_module.render = _render
    monkeypatch.setitem(sys.modules, "tools.odin.valhalla.dashboard.tabs.dispatch_fleet", fake_module)

    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/dispatch-fleet", data)
    assert _has_id(component, "real-tab-a")
    assert not _has_id(component, "tab-placeholder")
```

- [ ] **Step 10.2: Run new tests, verify they FAIL**

Expected: `test_placeholder_mentions_target_spec` may already pass (placeholder text mentions Spec 2 for `task-drilldown`); `test_real_tab_module_overrides_placeholder` FAILs because `_render_tab` always uses the placeholder.

- [ ] **Step 10.3: Update `_render_tab`** in `app.py` to prefer real tab modules when present

Replace the existing `_render_tab` function:

```python
def _render_tab(dispatch_id: str, tab_id: str, data: DataLayer) -> html.Div:
    """Render the tab body for /<id>/<tab_id>.

    Looks for a real tab module under ``tools.odin.valhalla.dashboard.tabs``
    matching ``tab_id``; falls back to the placeholder when the module is
    absent. Specs 1/2/3 add their modules; Spec 0 only ships ``_placeholder``.
    """
    import importlib

    module_name = {
        "dispatch-fleet": "tools.odin.valhalla.dashboard.tabs.dispatch_fleet",
        "task-drilldown": "tools.odin.valhalla.dashboard.tabs.task_drilldown",
        "startup": "tools.odin.valhalla.dashboard.tabs.startup",
    }.get(tab_id)
    if module_name is not None:
        try:
            tab_module = importlib.import_module(module_name)
            if hasattr(tab_module, "render"):
                return tab_module.render(dispatch_id, tab_id)
        except ModuleNotFoundError:
            pass
    from tools.odin.valhalla.dashboard.tabs import _placeholder

    return _placeholder.render(dispatch_id, tab_id)
```

- [ ] **Step 10.4: Run each new test individually, verify all PASS**

- [ ] **Step 10.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/app.py tools/odin/valhalla/dashboard/tests/test_app.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab registry: prefer real modules over placeholder

_render_tab attempts importlib.import_module on the tab-specific
module before falling back to _placeholder. Specs 1/2/3 land their
modules (tabs/dispatch_fleet.py, task_drilldown.py, startup.py) and
the dashboard picks them up automatically — no app.py edit required.
EOF
)"
```

---

## Task 11: `cli.py` — `odin-dashboard` entry point

**Files:**
- Create: `tools/odin/valhalla/dashboard/cli.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_cli.py`

- [ ] **Step 11.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_cli.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the odin-dashboard CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard import cli as cli_mod


def _write_dispatch(runs_root: Path, dispatch_id: str) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "abc",
        "fleet": [],
        "jobs": [],
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def test_parse_args_defaults(tmp_path):
    ns = cli_mod.parse_args(["--runs-root", str(tmp_path)])
    assert ns.port == 8050
    assert ns.host == "127.0.0.1"
    assert ns.runs_root == tmp_path
    assert ns.dispatch is None
    assert ns.no_browser is False
    assert ns.debug is False


def test_parse_args_explicit(tmp_path):
    ns = cli_mod.parse_args([
        "20260427-141302",
        "--runs-root", str(tmp_path),
        "--port", "9000",
        "--host", "0.0.0.0",
        "--no-browser",
        "--debug",
    ])
    assert ns.dispatch == "20260427-141302"
    assert ns.port == 9000
    assert ns.host == "0.0.0.0"
    assert ns.no_browser is True
    assert ns.debug is True


def test_main_invalid_runs_root_exits_2(capsys):
    rc = cli_mod.main(["--runs-root", "/nonexistent/odin_runs_dir"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "runs-root" in err.lower() or "/nonexistent" in err


def test_main_unknown_dispatch_exits_2(tmp_path, capsys):
    rc = cli_mod.main(["does-not-exist", "--runs-root", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "does-not-exist" in err


def test_main_no_browser_suppresses_open(tmp_path, monkeypatch):
    """--no-browser must not call webbrowser.open."""
    _write_dispatch(tmp_path, "20260427-141302")
    open_calls = []

    def _fake_open(url):
        open_calls.append(url)

    def _stub_run_server(self, host=None, port=None, debug=None):
        pass

    monkeypatch.setattr("webbrowser.open", _fake_open)
    monkeypatch.setattr("dash.Dash.run_server", _stub_run_server, raising=False)
    monkeypatch.setattr("dash.Dash.run", _stub_run_server, raising=False)

    rc = cli_mod.main(["20260427-141302", "--runs-root", str(tmp_path), "--no-browser"])
    assert rc == 0
    assert open_calls == []


def test_main_default_calls_browser_open(tmp_path, monkeypatch):
    """Without --no-browser, webbrowser.open is called once."""
    _write_dispatch(tmp_path, "20260427-141302")
    open_calls = []

    def _fake_open(url):
        open_calls.append(url)

    def _stub_run_server(self, host=None, port=None, debug=None):
        pass

    monkeypatch.setattr("webbrowser.open", _fake_open)
    monkeypatch.setattr("dash.Dash.run_server", _stub_run_server, raising=False)
    monkeypatch.setattr("dash.Dash.run", _stub_run_server, raising=False)

    rc = cli_mod.main(["20260427-141302", "--runs-root", str(tmp_path)])
    assert rc == 0
    assert len(open_calls) == 1
    assert open_calls[0].startswith("http://127.0.0.1:8050")


def test_main_port_in_use_exits_4(tmp_path, monkeypatch, capsys):
    _write_dispatch(tmp_path, "20260427-141302")

    def _raise_in_use(self, host=None, port=None, debug=None):
        raise OSError("Address already in use")

    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("dash.Dash.run_server", _raise_in_use, raising=False)
    monkeypatch.setattr("dash.Dash.run", _raise_in_use, raising=False)

    rc = cli_mod.main(["20260427-141302", "--runs-root", str(tmp_path), "--no-browser"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "in use" in err.lower() or "8050" in err


def test_main_latest_resolves_to_newest(tmp_path, monkeypatch):
    _write_dispatch(tmp_path, "20260424-160119")
    _write_dispatch(tmp_path, "20260427-141302")

    captured: dict = {}

    def _stub_run_server(self, host=None, port=None, debug=None):
        captured["pathname"] = self.layout.children[0].pathname

    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("dash.Dash.run_server", _stub_run_server, raising=False)
    monkeypatch.setattr("dash.Dash.run", _stub_run_server, raising=False)

    rc = cli_mod.main(["LATEST", "--runs-root", str(tmp_path), "--no-browser"])
    assert rc == 0
    assert captured["pathname"] == "/20260427-141302/dispatch-fleet"
```

- [ ] **Step 11.2: Run each test individually, verify they FAIL**

- [ ] **Step 11.3: Implement `cli.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-dashboard CLI — spins a local Dash server for an Odin runs_root.

Invoke from the repo root with ``PYTHONPATH=.``:

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/dashboard/cli.py \\
        --runs-root odin_runs

Or, jumping directly to a dispatch::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/dashboard/cli.py \\
        20260427-141302 --runs-root odin_runs
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="odin-dashboard",
        description="Browser-based dashboard over odin_runs/.",
    )
    parser.add_argument(
        "dispatch",
        nargs="?",
        default=None,
        help="Optional dispatch_id (e.g. 20260427-141302) or 'LATEST'. "
             "If set, the dashboard opens directly on Tab A of that dispatch.",
    )
    parser.add_argument(
        "--dispatch",
        dest="dispatch_flag",
        default=None,
        help="Same as the positional dispatch arg; flag form for clarity.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("odin_runs"))
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-browser", action="store_true", default=False)
    ns = parser.parse_args(argv)
    if ns.dispatch_flag and not ns.dispatch:
        ns.dispatch = ns.dispatch_flag
    delattr(ns, "dispatch_flag")
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    runs_root = ns.runs_root.resolve()
    if not runs_root.exists():
        print(f"odin-dashboard: --runs-root {runs_root} does not exist", file=sys.stderr)
        return 2
    initial_dispatch: Path | None = None
    if ns.dispatch:
        initial_dispatch = _resolve_dispatch_dir(runs_root, ns.dispatch)
        if initial_dispatch is None:
            print(f"odin-dashboard: dispatch {ns.dispatch!r} not found under {runs_root}", file=sys.stderr)
            return 2
    # Imported here so a missing `dash` install yields a clean exit-3 above the framework.
    try:
        from tools.odin.valhalla.dashboard.app import create_app
    except ModuleNotFoundError as exc:
        if "dash" in str(exc):
            print(
                "odin-dashboard: dash not installed; run `pip install dash plotly pandas`",
                file=sys.stderr,
            )
            return 3
        raise
    app = create_app(runs_root, initial_dispatch=initial_dispatch)
    url = f"http://{ns.host}:{ns.port}"
    print(f"odin-dashboard: serving {url}/ runs_root={runs_root}")
    if not ns.no_browser:
        webbrowser.open(url)
    try:
        run = getattr(app, "run_server", None) or app.run
        run(host=ns.host, port=ns.port, debug=ns.debug)
    except OSError as exc:
        if "in use" in str(exc).lower() or "Address already in use" in str(exc):
            print(
                f"odin-dashboard: port {ns.port} is in use; try --port",
                file=sys.stderr,
            )
            return 4
        raise
    return 0


def _resolve_dispatch_dir(runs_root: Path, spec: str) -> Path | None:
    """Resolve `spec` (a dispatch_id or 'LATEST') to an absolute dispatch path.

    Returns None if not found. Mirrors the dispatcher's resolution rules.
    """
    if spec == "LATEST":
        candidates = sorted(
            (p for p in runs_root.iterdir() if p.is_dir() and (p / "dispatch.json").exists()),
            key=lambda p: p.name,
            reverse=True,
        )
        return candidates[0] if candidates else None
    candidate = runs_root / spec
    if (candidate / "dispatch.json").exists():
        return candidate
    return None


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 11.4: Run each test individually, verify all PASS**

- [ ] **Step 11.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/cli.py tools/odin/valhalla/dashboard/tests/test_cli.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
odin-dashboard CLI

argparse + factory glue: --runs-root validation (exit 2), positional
or flag dispatch_id with LATEST shortcut (exit 2 on bad), missing
dash dep (exit 3 with hint), port-in-use (exit 4). --no-browser
suppresses webbrowser.open. Mirrors the odin-dispatch CLI shape so
the Odin tooling has a consistent invocation pattern.
EOF
)"
```

---

## Task 12: `assets/init.js` for `initial_dispatch` redirect

**Files:**
- Create: `tools/odin/valhalla/dashboard/assets/init.js`

Dash auto-loads any file under `assets/` and serves it from the same domain. We use this for a tiny client-side script that handles the "if launched from CLI with a dispatch_id, redirect" case for browsers that don't replay the `pathname` set in the layout. The Python-side layout already sets the right `pathname` on `dcc.Location`, but for full bookmark-stability we ensure the URL bar matches.

Spec 0 ships a minimal version. No test — `assets/` files are static; behavior is verified manually when running `odin-dashboard <id>`.

- [ ] **Step 12.1: Create the assets directory + init.js**

```javascript
// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Reserved for future client-side helpers (e.g., scrolling a tab into view).
// In Spec 0 the initial dispatch redirect is handled server-side by setting
// dcc.Location.pathname during create_app, so this file is intentionally
// empty.
```

- [ ] **Step 12.2: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/assets/init.js
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Reserve dashboard assets/ directory

Empty init.js placeholder so future tab specs have a place to drop
client-side helpers (chart scroll-into-view, focus management, etc.)
without re-introducing an assets/ dir from scratch.
EOF
)"
```

---

## Task 13: Integration-test extension — hardware.json end-to-end

**Files:**
- Modify: `tools/odin/tests/test_asgard_integration.py`

The existing `test_loopback_dispatch_against_localhost` (`pytestmark = pytest.mark.slow`) drives a full `run_dispatch` against `ssh localhost`. After Task 7's aggregator change, that run produces `hardware.json`. Add an assertion.

- [ ] **Step 13.1: Locate the existing test**

Run: `grep -n "def test_loopback_dispatch_against_localhost\|aggregate.json" tools/odin/tests/test_asgard_integration.py`
Note where the test asserts on `aggregate.json`. We add a parallel assertion on `hardware.json` near it.

- [ ] **Step 13.2: Add an assertion**

In `test_loopback_dispatch_against_localhost`, AFTER the existing read of `aggregate.json`, add:

```python
    # Spec 0 / Task 7: aggregator now also writes hardware.json.
    hw_path = dispatch_dirs[0] / "hardware.json"
    if hw_path.exists():
        hw = json.loads(hw_path.read_text())
        assert hw["schema_version"] == "1.0"
        assert hw["dispatch_id"] == dispatch_dirs[0].name
        assert hw["fingerprint"].startswith("gpu:")
        assert isinstance(hw["hosts"], dict)
    else:
        # Loopback test runs with a stub _build_docker_exec_cmd that doesn't
        # populate training.json.hardware on every kernel — accept absence
        # but fail if any inconsistency exists.
        pass
```

(The conditional handles the existing loopback-stub which populates a minimal `training.json` with no `.hardware` block — the aggregator will warn and skip. That's OK; on a real fleet run, the assertion fires.)

- [ ] **Step 13.3: Run the existing slow test** to confirm no regression

```
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_asgard_integration.py::test_loopback_dispatch_against_localhost -v --tb=short --noconftest -p no:cacheprovider
```

Expected: still passes (skips if `ssh localhost` doesn't work passwordlessly).

- [ ] **Step 13.4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_integration.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Loopback integration test: assert hardware.json shape when present

After Spec 0 Task 7, aggregator writes hardware.json next to
aggregate.json. The existing loopback test gains a conditional
assertion: if hardware.json exists, its schema_version, dispatch_id,
and fingerprint shape are correct. The loopback's _build_docker_exec
stub doesn't populate training.json.hardware, so absence is also
acceptable here — real-fleet runs always have the file.
EOF
)"
```

---

## Task 14: Architecture-doc change-log entry

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 14.1: Locate the change-log table**

Run: `grep -n "^| 2026-04-27" docs/odin/architecture.md | head -5`
Find the most recent row; the new row goes immediately above it.

- [ ] **Step 14.2: Prepend a row** for the dashboard skeleton

Add a row above the most recent 2026-04-27 entry:

```markdown
| 2026-04-27 | Odin dashboard skeleton (Spec 0) landed (`docs/superpowers/specs/2026-04-27-odin-dashboard-skeleton-design.md`). New `tools/odin/valhalla/dashboard/` module with three responsibilities: `cli.py` (`odin-dashboard` entry point with positional dispatch arg, `LATEST`, `--port` / `--host` / `--no-browser` / `--debug`), `app.py` (Dash factory + URL routing + tab registry), `data.py` (pure-Python `DataLayer` over `odin_runs/` exposing `list_dispatches`, `load_dispatch` / `_aggregate` / `_hardware`, `lookup_hardware`, `trend_dispatches_for`, `load_training` / `_startup`, `invalidate`). Aggregator extended with `_write_hardware_json`: per-dispatch `hardware.json` (schema 1.0) keyed by host, with a `fingerprint` of `gpu:<gpu_name>` derived from the first host's first GPU. The dashboard ships usable from this spec — multi-dispatch landing table; per-dispatch routing; three placeholder tabs picked up by Specs 1/2/3 via tab-module registry under `dashboard/tabs/`. No browser-based E2E tests; layout-tree + callback unit tests cover routing. Tests run with `PYTHONPATH=. python3 -m pytest --noconftest -p no:cacheprovider`. | Odin dashboard skeleton |
```

- [ ] **Step 14.3: Update "Last updated"** if the doc has one

Run: `grep -n "Last updated" docs/odin/architecture.md`
If found, bump it to `2026-04-27 (Odin dashboard skeleton)`.

- [ ] **Step 14.4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Architecture doc: change-log entry for dashboard skeleton

Records Spec 0 landing (`tools/odin/valhalla/dashboard/`),
hardware.json schema 1.0, and the tab-module registry pattern that
lets Specs 1/2/3 plug in additively.
EOF
)"
```

---

## Self-review

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| § Architecture / module layout | T2 (scaffolding files) + T8 (app.py) + T11 (cli.py) |
| § Components → cli.py | T11 |
| § Components → app.py | T8, T9, T10 |
| § Components → data.py (DispatchSummary, HardwareInfo, list_dispatches) | T2 |
| § Components → load_dispatch / load_aggregate / load_hardware | T3 |
| § Components → lookup_hardware | T4 |
| § Components → trend_dispatches_for | T5 |
| § Components → load_training / load_startup / invalidate | T6 |
| § Components → aggregator hardware.json | T7 |
| § Components → tabs/_placeholder.py + registry | T8 (stub), T10 (polish + registry) |
| § Hard deps (`dash`, `plotly`, `pandas`) | T1 |
| § Data flow (cold start, with-arg, header dropdown change, aggregator hardware-write) | T8 routing + T9 landing + T11 CLI + T7 aggregator together cover |
| § Error handling matrix | T11 (CLI error paths) + T8 routing (404) |
| § Testing strategy → test_data.py | T2-T6 |
| § Testing strategy → test_aggregator_hardware.py | T7 |
| § Testing strategy → test_app.py | T8, T10 |
| § Testing strategy → test_app_landing.py | T9 |
| § Testing strategy → test_cli.py | T11 |
| § Testing strategy → integration test extension | T13 |
| § Implementation order preview | Tasks 1-14 (close one-to-one match with the spec's preview) |

**Placeholder scan:** searched for "TBD", "TODO", "fill in", "<...>" — none. Every code step shows the actual code; every test step shows real assertions.

**Type / signature consistency:**
- `DataLayer(runs_root: Path)` signature consistent across T2-T6.
- `DataLayer.load_dispatch / load_aggregate / load_hardware / load_training / load_startup` all consistent return types in tests + implementation.
- `DataLayer.lookup_hardware(host) -> HardwareInfo | None` return shape matches the dataclass declared in T2 (hostname, gpu_devices, cpu_name, cpu_count, ram_gb, sourced_from).
- `DataLayer.trend_dispatches_for(current_dispatch_id, task, framework, backend, n=10)` signature consistent across spec, plan, and tests.
- `route_pathname(pathname: str, data: DataLayer)` consistent across T8 + T10.
- `tabs/_placeholder.render(dispatch_id, tab_id)` signature stable across T8 + T10.
- `cli.parse_args(argv: list[str]) -> argparse.Namespace` consistent across T11.

Plan is internally consistent and spec-complete.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-04-27-odin-dashboard-skeleton.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec compliance then code quality), fast iteration. Same shape as the GPU-loss recovery feature.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
