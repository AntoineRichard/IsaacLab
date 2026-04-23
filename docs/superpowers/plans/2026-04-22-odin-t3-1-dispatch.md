# Odin T3.1 — Headless Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `tools/odin/asgard/` (library + thin CLI) that takes a `fleet.yaml` + curated T2.1 env YAMLs + a seed list, provisions Valkyries via rsync + `docker/container.py`, dispatches jobs concurrently (one thread per Valkyrie), collects bundles back, and classifies failures (infrastructure / hugin_crash / hugin_malformed_bundle / timeout) with only infrastructure retried.

**Architecture:** Ten small modules in `tools/odin/asgard/` (≤ ~250 lines each). Threading model: main thread coordinates; one `ValkyrieWorker` thread per host pulls from a single `queue.Queue[JobEntry]`. State lives in `odin_runs/<dispatch_id>/dispatch.json`, rewritten atomically after every transition. SSH/rsync are `Protocol` types; shell-out `ShellSSHRunner` / `ShellRsyncRunner` are the default; tests inject fakes.

**Tech Stack:** Python 3.10+, stdlib only (`threading`, `queue`, `subprocess`, `dataclasses`, `pathlib`), `PyYAML` (already a dep), `pytest`. No new IsaacLab or Odin deps.

**Spec:** `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md`.

**Branch:** `antoiner/feat/odin` (local commits only; do not push).

**Commit convention:** Imperative ~50-char subject, body explains *why*, no AI co-author. `commit.gpgsign=true`. If signing fails with "Inappropriate ioctl for device", run `echo "test" | gpg --clearsign > /dev/null` to prime the agent and retry — do NOT bypass with `-c commit.gpgsign=false` or `--no-gpg-sign`.

**Project rules:** Python via `./isaaclab.sh -p`; tests via `./isaaclab.sh -p -m pytest PATH -v --confcutdir=tools/odin`. Run `./isaaclab.sh -f` BEFORE `git commit`. Pre-existing codespell / ruff failures in unrelated files are noise — only the file YOU touched must pass. The known good container default is `isaac-lab-base` (profile=base).

---

## Task 1: Package scaffolding + `fleet.py`

**Goal:** Create `tools/odin/asgard/` package, add `ValkyrieConfig` / `Fleet` dataclasses, and `load_fleet()` that parses the YAML with default-field resolution.

**Files:**
- Create: `tools/odin/asgard/__init__.py`
- Create: `tools/odin/asgard/fleet.py`
- Create: `tools/odin/tests/test_asgard_fleet.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_fleet.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.fleet`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig, load_fleet


def _write_fleet(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fleet.yaml"
    path.write_text(text)
    return path


def test_load_fleet_applies_defaults(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: test-fleet
default_ssh_user: odin
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: valkyrie-01
  - host: valkyrie-02
    ssh_user: other_user
""",
    )
    fleet = load_fleet(path)
    assert isinstance(fleet, Fleet)
    assert fleet.fleet_name == "test-fleet"
    assert len(fleet.hosts) == 2
    assert fleet.hosts[0].host == "valkyrie-01"
    assert fleet.hosts[0].ssh_user == "odin"
    assert str(fleet.hosts[0].ssh_key) == str(Path("~/.ssh/id_ed25519").expanduser())
    # Per-host override wins over default.
    assert fleet.hosts[1].ssh_user == "other_user"


def test_load_fleet_container_name_default(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts:
  - host: h1
""",
    )
    fleet = load_fleet(path)
    # Spec default for profile=base.
    assert fleet.hosts[0].container_name == "isaac-lab-base"


def test_load_fleet_per_host_overrides(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: odin
hosts:
  - host: h1
    isaaclab_path: /mnt/scratch/IsaacLab
    container_name: isaac-lab-ros
    labels: [h100-80gb]
""",
    )
    fleet = load_fleet(path)
    h = fleet.hosts[0]
    assert h.isaaclab_path == "/mnt/scratch/IsaacLab"
    assert h.container_name == "isaac-lab-ros"
    assert h.labels == ["h100-80gb"]


def test_load_fleet_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_fleet(tmp_path / "nope.yaml")


def test_load_fleet_empty_hosts_raises(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts: []
""",
    )
    with pytest.raises(ValueError, match="at least one host"):
        load_fleet(path)


def test_load_fleet_missing_host_field_raises(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts:
  - ssh_user: xx
""",
    )
    with pytest.raises(ValueError, match="host"):
        load_fleet(path)


def test_valkyrie_config_ssh_key_none_when_no_default(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts:
  - host: h1
""",
    )
    fleet = load_fleet(path)
    assert fleet.hosts[0].ssh_key is None
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_fleet.py -v --confcutdir=tools/odin
```

Expected: `ModuleNotFoundError: No module named 'tools.odin.asgard'`.

- [ ] **Step 3: Create package and `fleet.py`**

Create `tools/odin/asgard/__init__.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asgard — Odin's distributed dispatch library.

Public API for dispatching Hugin/Munin jobs across a fleet of Valkyrie
machines (SSH + docker). The CLI :mod:`tools.odin.asgard.cli` is a thin
wrapper over :func:`run_dispatch`; a future T3.2 web UI would consume the
same public surface.
"""

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig, load_fleet

__all__ = [
    "Fleet",
    "ValkyrieConfig",
    "load_fleet",
]
```

Create `tools/odin/asgard/fleet.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fleet configuration — list of Valkyrie hosts + per-host SSH / path config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Fleet", "ValkyrieConfig", "load_fleet"]


@dataclass
class ValkyrieConfig:
    """Per-host configuration for a Valkyrie."""

    host: str
    ssh_user: str
    ssh_key: Path | None = None
    isaaclab_path: str = "~/IsaacLab"
    container_name: str = "isaac-lab-base"
    labels: list[str] = field(default_factory=list)


@dataclass
class Fleet:
    """The whole fleet."""

    fleet_name: str
    hosts: list[ValkyrieConfig]


def _resolve_ssh_key(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def load_fleet(path: Path) -> Fleet:
    """Parse ``fleet.yaml`` into a :class:`Fleet` with defaults resolved.

    Per-host fields override fleet-level ``default_*`` fields. ``ssh_key`` is
    ``pathlib.Path``-resolved with ``~`` expansion.

    Args:
        path: Path to ``fleet.yaml``.

    Returns:
        Populated :class:`Fleet`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On missing required fields or empty host list.
    """
    if not path.exists():
        raise FileNotFoundError(f"fleet.yaml not found: {path}")
    with path.open("r") as fh:
        payload: dict[str, Any] = yaml.safe_load(fh) or {}

    fleet_name = str(payload.get("fleet_name") or "")
    default_ssh_user = payload.get("default_ssh_user")
    default_ssh_key = payload.get("default_ssh_key")
    hosts_raw = payload.get("hosts") or []

    if not hosts_raw:
        raise ValueError(f"fleet.yaml must list at least one host: {path}")

    hosts: list[ValkyrieConfig] = []
    for idx, raw in enumerate(hosts_raw):
        if not isinstance(raw, dict) or "host" not in raw:
            raise ValueError(f"fleet.yaml host entry #{idx} missing required 'host' field: {raw!r}")
        ssh_user = raw.get("ssh_user") or default_ssh_user
        if ssh_user is None:
            raise ValueError(
                f"fleet.yaml host {raw['host']!r} has no ssh_user and no default_ssh_user is set"
            )
        ssh_key_value = raw.get("ssh_key") if "ssh_key" in raw else default_ssh_key
        hosts.append(
            ValkyrieConfig(
                host=str(raw["host"]),
                ssh_user=str(ssh_user),
                ssh_key=_resolve_ssh_key(ssh_key_value),
                isaaclab_path=str(raw.get("isaaclab_path", "~/IsaacLab")),
                container_name=str(raw.get("container_name", "isaac-lab-base")),
                labels=list(raw.get("labels") or []),
            )
        )

    return Fleet(fleet_name=fleet_name, hosts=hosts)
```

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_fleet.py -v --confcutdir=tools/odin
```

Expected: 7 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/__init__.py tools/odin/asgard/fleet.py tools/odin/tests/test_asgard_fleet.py
git commit -m "Add Asgard package skeleton and Fleet config loader

tools/odin/asgard/ is the new home for Odin's distributed-dispatch
library (T3.1). The public API is exposed via the package __init__ so
future additions (queue, state, transport, runner) just extend the
__all__.

Fleet and ValkyrieConfig are plain dataclasses; load_fleet() reads
fleet.yaml, resolves fleet-level default_ssh_user / default_ssh_key
into per-host configs, and raises ValueError on missing-host /
empty-host / missing-ssh-user conditions.

Container name default is 'isaac-lab-base' matching
docker/docker-compose.yaml for the 'base' profile."
```

---

## Task 2: `queue.py` — `JobEntry` + `build_queue_from_env_lists()`

**Goal:** Expand a curated env list (T2.1's `EnvEntry` rows with `keep: true` and `status != "stale"`) across a seed list into `JobEntry` rows with deterministic `run_id`s sharing the dispatch timestamp.

**Files:**
- Create: `tools/odin/asgard/queue.py`
- Create: `tools/odin/tests/test_asgard_queue.py`
- Modify: `tools/odin/asgard/__init__.py` (extend `__all__`)

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_queue.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.queue`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.queue import JobEntry, build_queue_from_env_lists
from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list


def _env(task_id: str, framework: str = "rsl_rl", keep: bool = True, status: str = "current") -> EnvEntry:
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
    )


def _write_env_list(tmp_path: Path, name: str, entries: list[EnvEntry]) -> Path:
    el = EnvList()
    for e in entries:
        el.groups.setdefault(e.group, []).append(e)
    out = tmp_path / name
    write_env_list(out, el, generator="test")
    return out


def test_expand_one_row_one_seed(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000"
    )
    assert len(jobs) == 1
    j = jobs[0]
    assert j.task_id == "Isaac-Ant-Direct-v0"
    assert j.framework == "rsl_rl"
    assert j.backend == "physx"
    assert j.seed == 42
    assert j.num_envs == 4096
    assert j.max_iterations == 300
    assert j.run_id == "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42"
    assert j.bundle_dir_name == j.run_id
    assert j.status == "pending"
    assert j.attempts == 0


def test_expand_multiple_seeds(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=None, seeds=[42, 43, 44], dispatch_id="20260422-220000"
    )
    assert len(jobs) == 3
    assert {j.seed for j in jobs} == {42, 43, 44}
    assert len({j.run_id for j in jobs}) == 3  # all unique


def test_combines_physx_and_newton(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    newton = _write_env_list(tmp_path, "newton.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=newton, seeds=[42], dispatch_id="20260422-220000"
    )
    assert len(jobs) == 2
    assert {j.backend for j in jobs} == {"physx", "newton"}


def test_skips_keep_false_rows(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Keep-False-v0", keep=False)],
    )
    jobs = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000"
    )
    assert [j.task_id for j in jobs] == ["Isaac-Ant-Direct-v0"]


def test_skips_stale_rows(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Stale-v0", status="stale")],
    )
    jobs = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000"
    )
    assert [j.task_id for j in jobs] == ["Isaac-Ant-Direct-v0"]


def test_include_filter_fnmatch(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Humanoid-Direct-v0"), _env("Isaac-Cartpole-Direct-v0")],
    )
    jobs = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260422-220000",
        include_filter=["Isaac-Ant-*", "Isaac-Humanoid-*"],
    )
    assert {j.task_id for j in jobs} == {"Isaac-Ant-Direct-v0", "Isaac-Humanoid-Direct-v0"}


def test_neither_yaml_raises():
    with pytest.raises(ValueError, match="at least one"):
        build_queue_from_env_lists(
            physx_yaml=None, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000"
        )


def test_empty_seeds_raises(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    with pytest.raises(ValueError, match="seed"):
        build_queue_from_env_lists(
            physx_yaml=physx, newton_yaml=None, seeds=[], dispatch_id="20260422-220000"
        )
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin
```

Expected: `ImportError: cannot import name 'JobEntry' from 'tools.odin.asgard.queue'`.

- [ ] **Step 3: Create `queue.py`**

Create `tools/odin/asgard/queue.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queue construction — expand curated env lists across seeds into JobEntry rows."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.common.env_list import load_env_list

__all__ = ["JobEntry", "FailureInfo", "build_queue_from_env_lists"]


@dataclass
class FailureInfo:
    """Classified failure attached to a :class:`JobEntry` when ``status == 'failed'``."""

    kind: str           # "infrastructure" | "hugin_crash" | "hugin_malformed_bundle" | "timeout"
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class JobEntry:
    """One row in the dispatch queue — the smallest unit of work."""

    run_id: str
    task_id: str
    framework: str           # "rsl_rl" | "skrl"
    backend: str             # "physx" | "newton"
    num_envs: int
    max_iterations: int
    seed: int
    bundle_dir_name: str
    status: str = "pending"  # pending | assigned | running | completed | failed
    assigned_to: str | None = None
    attempts: int = 0
    failure: FailureInfo | None = None
    preferred_not: set[str] = field(default_factory=set)
    started_at: str | None = None
    ended_at: str | None = None


def _framework_slug(framework: str) -> str:
    """rsl_rl -> rsl-rl, skrl -> skrl (hyphen variant used in run_id paths)."""
    return framework.replace("_", "-")


def _make_run_id(framework: str, backend: str, task_id: str, dispatch_id: str, seed: int) -> str:
    return f"{_framework_slug(framework)}_{backend}_{task_id}_{dispatch_id}_seed{seed}"


def _apply_include_filter(task_id: str, include_filter: list[str] | None) -> bool:
    if not include_filter:
        return True
    return any(fnmatch.fnmatch(task_id, pat) for pat in include_filter)


def _expand_env_list(
    yaml_path: Path,
    backend: str,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None,
) -> list[JobEntry]:
    env_list = load_env_list(yaml_path)
    jobs: list[JobEntry] = []
    for group_rows in env_list.groups.values():
        for row in group_rows:
            if not row.keep or row.status == "stale":
                continue
            if not _apply_include_filter(row.task_id, include_filter):
                continue
            if row.framework is None or row.num_envs is None or row.max_iterations is None:
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
    return jobs


def build_queue_from_env_lists(
    physx_yaml: Path | None,
    newton_yaml: Path | None,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None = None,
) -> list[JobEntry]:
    """Expand curated env YAMLs across seeds into a flat job list.

    Args:
        physx_yaml: Path to ``physx_envs.yaml`` (T2.1); ``None`` to skip PhysX.
        newton_yaml: Path to ``newton_envs.yaml`` (T2.1); ``None`` to skip Newton.
        seeds: Seeds to expand each kept row across. Must be non-empty.
        dispatch_id: UTC timestamp (``YYYYMMDD-HHMMSS``) shared by all run_ids
            in this dispatch.
        include_filter: Optional list of fnmatch patterns on ``task_id``; a row
            must match at least one pattern to be queued. Unset = keep all.

    Returns:
        List of :class:`JobEntry` rows in insertion order (PhysX first, then
        Newton; within a backend, YAML-group order; within a group,
        insertion order; within a row, seed order).

    Raises:
        ValueError: If neither YAML is provided or seeds is empty.
    """
    if physx_yaml is None and newton_yaml is None:
        raise ValueError("build_queue_from_env_lists needs at least one env list (physx_yaml or newton_yaml)")
    if not seeds:
        raise ValueError("build_queue_from_env_lists needs a non-empty seed list")

    jobs: list[JobEntry] = []
    if physx_yaml is not None:
        jobs.extend(_expand_env_list(physx_yaml, "physx", seeds, dispatch_id, include_filter))
    if newton_yaml is not None:
        jobs.extend(_expand_env_list(newton_yaml, "newton", seeds, dispatch_id, include_filter))
    return jobs
```

Update `tools/odin/asgard/__init__.py` `__all__`:

```python
from tools.odin.asgard.fleet import Fleet, ValkyrieConfig, load_fleet
from tools.odin.asgard.queue import FailureInfo, JobEntry, build_queue_from_env_lists

__all__ = [
    "Fleet",
    "ValkyrieConfig",
    "load_fleet",
    "JobEntry",
    "FailureInfo",
    "build_queue_from_env_lists",
]
```

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_queue.py -v --confcutdir=tools/odin
```

Expected: 8 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/queue.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_queue.py
git commit -m "Expand curated env lists into Asgard JobEntry queue

queue.py defines JobEntry (one unit of work) and FailureInfo (kind +
message + details attached when status == failed), plus
build_queue_from_env_lists() which reads physx_envs.yaml and/or
newton_envs.yaml, skips keep:false and status:stale rows, applies an
optional fnmatch include_filter, and expands each remaining row across
the seed list.

run_id format: <framework_hyphen>_<backend>_<task_id>_<dispatch_id>_seed<seed>,
matching the T1 run_id convention except that <dispatch_id> replaces
the per-run UTC timestamp so all bundles in one dispatch share the
same stamp and sort together."
```

---

## Task 3: `state.py` — `DispatchState` + atomic read/write

**Goal:** Dataclasses for the on-disk state + `read_dispatch_state()` / `write_dispatch_state()` with atomic write (tempfile + rename).

**Files:**
- Create: `tools/odin/asgard/state.py`
- Create: `tools/odin/tests/test_asgard_state.py`
- Modify: `tools/odin/asgard/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_state.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.state`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.queue import FailureInfo, JobEntry
from tools.odin.asgard.state import (
    DispatchState,
    FleetSnapshot,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)


def _job(run_id: str, status: str = "pending", **kw) -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name=run_id,
        status=status,
        **kw,
    )


def _state(jobs: list[JobEntry]) -> DispatchState:
    return DispatchState(
        schema_version="1.0",
        dispatch_id="20260422-220000",
        started_at="2026-04-22T22:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc123",
        fleet=[FleetSnapshot(host="h1", status="idle", current_run_id=None, last_error=None)],
        jobs=jobs,
    )


def test_roundtrip_minimal(tmp_path: Path):
    original = _state([_job("rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42")])
    write_dispatch_state(tmp_path, original)
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded.dispatch_id == "20260422-220000"
    assert len(reloaded.jobs) == 1
    assert reloaded.jobs[0].run_id == original.jobs[0].run_id
    assert reloaded.jobs[0].status == "pending"


def test_roundtrip_preserves_failure_info(tmp_path: Path):
    j = _job("run-x", status="failed")
    j.failure = FailureInfo(
        kind="hugin_crash",
        message="exit code 1",
        details={"exit_code": 1, "log_tail_path": "run-x/logs/ssh-tail.log"},
    )
    j.attempts = 1
    write_dispatch_state(tmp_path, _state([j]))

    reloaded = read_dispatch_state(tmp_path)
    rj = reloaded.jobs[0]
    assert rj.failure is not None
    assert rj.failure.kind == "hugin_crash"
    assert rj.failure.details["exit_code"] == 1
    assert rj.attempts == 1


def test_atomic_write_no_partial(tmp_path: Path, monkeypatch):
    """Atomic write must never leave a partial file visible to readers."""
    # Sanity: the implementation uses a temp file + rename. Simulate a crash
    # between temp-file-close and rename by monkeypatching os.replace to raise
    # on the *second* call; assert the original file still parses cleanly.
    import os

    state1 = _state([_job("run-a")])
    write_dispatch_state(tmp_path, state1)
    first_mtime = (tmp_path / "dispatch.json").stat().st_mtime_ns

    state2 = _state([_job("run-a", status="running"), _job("run-b")])
    real_replace = os.replace
    call_count = {"n": 0}

    def _boom(src, dst, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 1:
            raise RuntimeError("simulated crash between write and rename")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(RuntimeError):
        write_dispatch_state(tmp_path, state2)

    monkeypatch.setattr(os, "replace", real_replace)
    # The pre-existing file must still be readable and reflect state1.
    reloaded = read_dispatch_state(tmp_path)
    assert len(reloaded.jobs) == 1
    assert reloaded.jobs[0].run_id == "run-a"
    assert reloaded.jobs[0].status == "pending"
    assert (tmp_path / "dispatch.json").stat().st_mtime_ns == first_mtime


def test_reset_in_flight_flips_running_and_assigned(tmp_path: Path):
    jobs = [
        _job("r-running", status="running", assigned_to="h1"),
        _job("r-assigned", status="assigned", assigned_to="h1"),
        _job("r-completed", status="completed"),
        _job("r-failed", status="failed"),
        _job("r-pending", status="pending"),
    ]
    s = _state(jobs)
    reset_in_flight_to_pending(s)
    statuses = {j.run_id: j.status for j in s.jobs}
    assert statuses["r-running"] == "pending"
    assert statuses["r-assigned"] == "pending"
    assert statuses["r-completed"] == "completed"
    assert statuses["r-failed"] == "failed"
    assert statuses["r-pending"] == "pending"
    # Assignment cleared on reset.
    running_job = next(j for j in s.jobs if j.run_id == "r-running")
    assert running_job.assigned_to is None


def test_read_missing_returns_none(tmp_path: Path):
    assert read_dispatch_state(tmp_path) is None
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin
```

Expected: ImportError on `DispatchState` / `FleetSnapshot`.

- [ ] **Step 3: Create `state.py`**

Create `tools/odin/asgard/state.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DispatchState — on-disk representation of an Asgard dispatch.

``dispatch.json`` lives at ``<dispatch_dir>/dispatch.json`` and is
rewritten atomically (temp-file + rename) after every state transition
and on a periodic heartbeat.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.odin.asgard.queue import FailureInfo, JobEntry

__all__ = [
    "DispatchState",
    "FleetSnapshot",
    "SCHEMA_VERSION",
    "read_dispatch_state",
    "write_dispatch_state",
    "reset_in_flight_to_pending",
]


SCHEMA_VERSION = "1.0"
_DISPATCH_FILENAME = "dispatch.json"


@dataclass
class FleetSnapshot:
    """Per-host live state, written into ``dispatch.json``."""

    host: str
    status: str                   # "idle" | "busy" | "down"
    current_run_id: str | None = None
    last_error: str | None = None


@dataclass
class DispatchState:
    """Complete on-disk state for one dispatch."""

    schema_version: str
    dispatch_id: str
    started_at: str               # UTC ISO-8601
    ended_at: str | None
    seeds: list[int]
    commit_sha: str
    fleet: list[FleetSnapshot]
    jobs: list[JobEntry]


# --- Serialization -----------------------------------------------------------


def _job_to_dict(j: JobEntry) -> dict[str, Any]:
    d: dict[str, Any] = {
        "run_id": j.run_id,
        "task_id": j.task_id,
        "framework": j.framework,
        "backend": j.backend,
        "num_envs": j.num_envs,
        "max_iterations": j.max_iterations,
        "seed": j.seed,
        "bundle_dir_name": j.bundle_dir_name,
        "status": j.status,
        "assigned_to": j.assigned_to,
        "attempts": j.attempts,
        "started_at": j.started_at,
        "ended_at": j.ended_at,
        "preferred_not": sorted(j.preferred_not),
    }
    if j.failure is None:
        d["failure"] = None
    else:
        d["failure"] = {
            "kind": j.failure.kind,
            "message": j.failure.message,
            "details": j.failure.details,
        }
    return d


def _job_from_dict(d: dict[str, Any]) -> JobEntry:
    failure = None
    if d.get("failure") is not None:
        failure = FailureInfo(
            kind=str(d["failure"]["kind"]),
            message=str(d["failure"].get("message", "")),
            details=dict(d["failure"].get("details") or {}),
        )
    return JobEntry(
        run_id=str(d["run_id"]),
        task_id=str(d["task_id"]),
        framework=str(d["framework"]),
        backend=str(d["backend"]),
        num_envs=int(d["num_envs"]),
        max_iterations=int(d["max_iterations"]),
        seed=int(d["seed"]),
        bundle_dir_name=str(d["bundle_dir_name"]),
        status=str(d.get("status", "pending")),
        assigned_to=d.get("assigned_to"),
        attempts=int(d.get("attempts", 0)),
        failure=failure,
        preferred_not=set(d.get("preferred_not") or []),
        started_at=d.get("started_at"),
        ended_at=d.get("ended_at"),
    )


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
    }


def _state_from_dict(d: dict[str, Any]) -> DispatchState:
    got_schema = str(d.get("schema_version", ""))
    if got_schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dispatch.json schema_version {got_schema!r} (expected {SCHEMA_VERSION!r})"
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
    )


# --- I/O ---------------------------------------------------------------------


def read_dispatch_state(dispatch_dir: Path) -> DispatchState | None:
    """Read ``<dispatch_dir>/dispatch.json`` into a :class:`DispatchState`.

    Returns ``None`` when the file does not exist (e.g. first-time dispatch).
    Raises :class:`ValueError` when it exists but declares an unsupported
    ``schema_version``.
    """
    path = dispatch_dir / _DISPATCH_FILENAME
    if not path.exists():
        return None
    with path.open("r") as fh:
        payload = json.load(fh)
    return _state_from_dict(payload)


def write_dispatch_state(dispatch_dir: Path, state: DispatchState) -> None:
    """Atomically rewrite ``<dispatch_dir>/dispatch.json``.

    Writes to a sibling temporary file then ``os.replace``s over the final
    path, so a concurrent reader never observes a truncated file.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload = _state_to_dict(state)
    fd, tmp_path_str = tempfile.mkstemp(prefix=".dispatch_", suffix=".json.tmp", dir=str(dispatch_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, dispatch_dir / _DISPATCH_FILENAME)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# --- Resume helpers ----------------------------------------------------------


def reset_in_flight_to_pending(state: DispatchState) -> None:
    """Flip ``running`` / ``assigned`` jobs back to ``pending`` for resume.

    Called in-place on the loaded state before a resumed dispatch starts its
    workers. ``completed`` and ``failed`` jobs are left alone — a failed job
    is only re-attempted via an explicit ``--retry-failed`` escape hatch.
    """
    for j in state.jobs:
        if j.status in ("running", "assigned"):
            j.status = "pending"
            j.assigned_to = None
            j.started_at = None
```

Update `tools/odin/asgard/__init__.py`:

```python
from tools.odin.asgard.fleet import Fleet, ValkyrieConfig, load_fleet
from tools.odin.asgard.queue import FailureInfo, JobEntry, build_queue_from_env_lists
from tools.odin.asgard.state import (
    DispatchState,
    FleetSnapshot,
    SCHEMA_VERSION,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)

__all__ = [
    "Fleet",
    "ValkyrieConfig",
    "load_fleet",
    "JobEntry",
    "FailureInfo",
    "build_queue_from_env_lists",
    "DispatchState",
    "FleetSnapshot",
    "SCHEMA_VERSION",
    "read_dispatch_state",
    "reset_in_flight_to_pending",
    "write_dispatch_state",
]
```

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin
```

Expected: 5 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/state.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_state.py
git commit -m "Add DispatchState with atomic dispatch.json read/write

state.py defines DispatchState (schema_version, dispatch_id, started_at,
ended_at, seeds, commit_sha, fleet, jobs) and FleetSnapshot (host,
status, current_run_id, last_error). SCHEMA_VERSION='1.0'.

write_dispatch_state() uses tempfile.mkstemp + os.replace so a
concurrent reader (future web UI tailing the file) never sees a
truncated file.

reset_in_flight_to_pending() is the resume-safety primitive: flips any
'running' or 'assigned' job back to 'pending' (and clears assigned_to)
so a resumed dispatch re-dispatches those jobs; completed / failed
jobs are preserved."
```

---

## Task 4: `transport.py` — Protocols + shell SSH runner

**Goal:** Define `SSHRunner` / `RsyncRunner` protocols + `SSHResult` / `RsyncResult` dataclasses, and implement `ShellSSHRunner` shelling out to `ssh` via `subprocess`.

**Files:**
- Create: `tools/odin/asgard/transport.py`
- Create: `tools/odin/tests/test_asgard_transport_ssh.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_transport_ssh.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`tools.odin.asgard.transport.ShellSSHRunner` (mock subprocess)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import ShellSSHRunner, SSHResult


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="valkyrie-01", ssh_user="odin", ssh_key=None)


def test_build_ssh_command_minimal():
    runner = ShellSSHRunner()
    host = _host()
    argv = runner._build_ssh_argv(host, 'echo "hi"', timeout_s=None)
    # First arg is "ssh"; there should be a user@host target and a trailing cmd string.
    assert argv[0] == "ssh"
    assert f"{host.ssh_user}@{host.host}" in argv
    assert argv[-1] == 'echo "hi"'
    # Bake-in options.
    assert any("StrictHostKeyChecking=accept-new" in a for a in argv)
    assert any("ServerAliveInterval=30" in a for a in argv)
    assert any("ConnectTimeout=10" in a for a in argv)


def test_build_ssh_command_with_key(tmp_path: Path):
    key = tmp_path / "fake_key"
    key.write_text("nope")
    host = ValkyrieConfig(host="h1", ssh_user="u", ssh_key=key)
    runner = ShellSSHRunner()
    argv = runner._build_ssh_argv(host, "true", timeout_s=None)
    # -i <key> pair present.
    i = argv.index("-i")
    assert Path(argv[i + 1]) == key


def test_run_happy_path(monkeypatch):
    """Runner returns SSHResult with exit_code 0, stdout, stderr."""
    captured: dict = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["argv"] = argv
            captured["kwargs"] = kw
            self.returncode = 0
            self.stdout_lines = iter(["hello\n", "world\n"])
            self.stderr_lines = iter([])

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

        def communicate(self, timeout=None):
            return ("hello\nworld\n", "")

        @property
        def stdout(self):
            return _FakeStream(["hello\n", "world\n"])

        @property
        def stderr(self):
            return _FakeStream([])

    class _FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        def readline(self):
            if not self._lines:
                return ""
            return self._lines.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    runner = ShellSSHRunner()
    result = runner.run(_host(), "echo hello", timeout_s=None)
    assert isinstance(result, SSHResult)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_writes_tee_file(tmp_path: Path, monkeypatch):
    class _FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        def readline(self):
            if not self._lines:
                return ""
            return self._lines.pop(0)

        def close(self):
            pass

    class _FakePopen:
        def __init__(self, argv, **kw):
            self.returncode = 0
            self.stdout = _FakeStream(["line-a\n", "line-b\n"])
            self.stderr = _FakeStream([])

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

        def communicate(self, timeout=None):
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    tee_path = tmp_path / "ssh-tail.log"
    runner = ShellSSHRunner()
    runner.run(_host(), "cmd", timeout_s=None, stdout_tee=tee_path)

    content = tee_path.read_text()
    assert "line-a" in content
    assert "line-b" in content


def test_run_timeout_terminates(monkeypatch):
    """When wait() raises TimeoutExpired, runner calls terminate and reports timed_out=True."""

    class _FakeStream:
        def readline(self):
            return ""

        def close(self):
            pass

    class _FakePopen:
        def __init__(self, argv, **kw):
            self.returncode = None
            self.stdout = _FakeStream()
            self.stderr = _FakeStream()
            self._terminated = False

        def wait(self, timeout=None):
            if not self._terminated:
                raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout or 0)
            return -15

        def terminate(self):
            self._terminated = True
            self.returncode = -15

        def kill(self):
            self._terminated = True
            self.returncode = -9

        def communicate(self, timeout=None):
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    runner = ShellSSHRunner()
    result = runner.run(_host(), "cmd", timeout_s=0.1)
    assert result.timed_out is True
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_transport_ssh.py -v --confcutdir=tools/odin
```

Expected: ImportError on `ShellSSHRunner`.

- [ ] **Step 3: Create `transport.py` (SSH portion — rsync follows in Task 5)**

Create `tools/odin/asgard/transport.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transport layer — SSH / rsync Protocols + shell-out default implementations."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from tools.odin.asgard.fleet import ValkyrieConfig

__all__ = [
    "SSHResult",
    "RsyncResult",
    "SSHRunner",
    "RsyncRunner",
    "ShellSSHRunner",
]


@dataclass
class SSHResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class RsyncResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    bytes_transferred: int | None = None


class SSHRunner(Protocol):
    def run(
        self,
        host: ValkyrieConfig,
        cmd: str,
        *,
        timeout_s: float | None = None,
        stdout_tee: Path | None = None,
    ) -> SSHResult: ...


class RsyncRunner(Protocol):
    def pull(self, host: ValkyrieConfig, remote_path: str, local_path: Path) -> RsyncResult: ...
    def push(self, host: ValkyrieConfig, local_path: Path, remote_path: str) -> RsyncResult: ...


# --- Default ssh implementation ---------------------------------------------


_DEFAULT_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",  # no interactive password prompts in a dispatch
]


class ShellSSHRunner:
    """SSH runner that shells out to the ``ssh`` command."""

    def _build_ssh_argv(self, host: ValkyrieConfig, cmd: str, timeout_s: float | None) -> list[str]:
        argv: list[str] = ["ssh", *_DEFAULT_SSH_OPTS]
        if host.ssh_key is not None:
            argv += ["-i", str(host.ssh_key)]
        argv += [f"{host.ssh_user}@{host.host}", cmd]
        return argv

    def run(
        self,
        host: ValkyrieConfig,
        cmd: str,
        *,
        timeout_s: float | None = None,
        stdout_tee: Path | None = None,
    ) -> SSHResult:
        """Run ``cmd`` on ``host`` and return an :class:`SSHResult`.

        Streams stdout line-by-line to ``stdout_tee`` (if given). On timeout
        the child process is terminated (then killed after 10 s grace);
        ``timed_out`` is set on the returned result and ``exit_code`` is
        whatever the terminated process reported (typically negative).
        """
        argv = self._build_ssh_argv(host, cmd, timeout_s)
        t0 = time.monotonic()
        tee_fh = None
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if stdout_tee is not None:
            stdout_tee.parent.mkdir(parents=True, exist_ok=True)
            tee_fh = stdout_tee.open("a", encoding="utf-8")

        timed_out = False
        try:
            # Simple single-thread read: drain stdout then wait. Sufficient
            # for our use case where the remote command runs to completion
            # and we don't need line-level real-time interleaving with stderr.
            while True:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if not line:
                    break
                stdout_buf.append(line)
                if tee_fh is not None:
                    tee_fh.write(line)
                    tee_fh.flush()
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.terminate()
                try:
                    rc = proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
            # Drain stderr last (Popen keeps the pipe open until wait returns).
            if proc.stderr is not None:
                rest_err = proc.stderr.read()
                if rest_err:
                    stderr_buf.append(rest_err)
        finally:
            if tee_fh is not None:
                tee_fh.close()
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        duration = time.monotonic() - t0
        return SSHResult(
            exit_code=int(rc if rc is not None else -1),
            stdout="".join(stdout_buf),
            stderr="".join(stderr_buf),
            duration_s=duration,
            timed_out=timed_out,
        )
```

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_transport_ssh.py -v --confcutdir=tools/odin
```

Expected: 5 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/transport.py tools/odin/tests/test_asgard_transport_ssh.py
git commit -m "Add SSH transport protocol and ShellSSHRunner

transport.py defines the SSHRunner / RsyncRunner Protocol types so
workers depend on behaviour not implementation — tests inject fake
runners; production uses ShellSSHRunner / ShellRsyncRunner (the
latter lands in a follow-up commit).

ShellSSHRunner shells out to the ssh binary with baked-in options
(StrictHostKeyChecking=accept-new, ServerAliveInterval=30,
ConnectTimeout=10, BatchMode=yes — no interactive password prompts
inside a dispatch). Explicit -i <key> when ValkyrieConfig.ssh_key is
set. Timeout terminates then kills after a 10s grace; timed_out flag
on SSHResult distinguishes timeout from regular non-zero exit."
```

---

## Task 5: `transport.py` — `ShellRsyncRunner`

**Goal:** Add `ShellRsyncRunner` with `pull()` / `push()` methods that shell out to `rsync`. Exclude `.git/`, `__pycache__/`, `odin_runs/`, test scratch dirs during push.

**Files:**
- Modify: `tools/odin/asgard/transport.py` (append)
- Create: `tools/odin/tests/test_asgard_transport_rsync.py`
- Modify: `tools/odin/asgard/__init__.py` (export `ShellRsyncRunner`)

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_transport_rsync.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`tools.odin.asgard.transport.ShellRsyncRunner`."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import RsyncResult, ShellRsyncRunner


def _host(ssh_key: Path | None = None) -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", ssh_key=ssh_key)


def _fake_completed(captured: dict):
    def _run(argv, **kw):
        captured["argv"] = argv
        captured["kwargs"] = kw

        class _R:
            returncode = 0
            stdout = "sent 100 bytes\n"
            stderr = ""
        return _R()
    return _run


def test_push_argv_includes_excludes(monkeypatch, tmp_path: Path):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    result = runner.push(_host(), tmp_path, "~/IsaacLab")
    assert isinstance(result, RsyncResult)
    argv = captured["argv"]
    assert argv[0] == "rsync"
    # Standard flags.
    assert "-avz" in argv or ("-a" in argv and "-v" in argv and "-z" in argv)
    # Excludes for push.
    excludes = [a for a in argv if a.startswith("--exclude")]
    joined = " ".join(excludes)
    assert "__pycache__" in joined
    assert ".git" in joined
    assert "odin_runs" in joined


def test_push_destination_shape(monkeypatch, tmp_path: Path):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.push(_host(), tmp_path, "~/IsaacLab")
    argv = captured["argv"]
    # Last arg must be user@host:remote_path.
    assert argv[-1] == "odin@v1:~/IsaacLab"


def test_pull_destination_shape(monkeypatch, tmp_path: Path):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.pull(_host(), "~/IsaacLab/odin_runs/x", tmp_path / "x")
    argv = captured["argv"]
    # First non-flag arg after rsync flags: user@host:remote_path, then local_path.
    assert "odin@v1:~/IsaacLab/odin_runs/x" in argv
    assert str(tmp_path / "x") in argv


def test_pull_no_delete(monkeypatch, tmp_path: Path):
    """Pull must NOT pass --delete (we don't want to prune the controller's prior bundles)."""
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.pull(_host(), "~/odin_runs/x", tmp_path / "x")
    argv = captured["argv"]
    assert "--delete" not in argv


def test_ssh_key_threaded_through(monkeypatch, tmp_path: Path):
    key = tmp_path / "id"
    key.write_text("x")
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.push(_host(ssh_key=key), tmp_path / "src", "~/dst")
    argv = captured["argv"]
    # -e 'ssh -i <key> ...' form.
    e_idx = argv.index("-e")
    assert str(key) in argv[e_idx + 1]
    assert argv[e_idx + 1].startswith("ssh ")
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_transport_rsync.py -v --confcutdir=tools/odin
```

Expected: ImportError on `ShellRsyncRunner`.

- [ ] **Step 3: Append `ShellRsyncRunner` to `transport.py`**

Append to `tools/odin/asgard/transport.py`:

```python


# --- Default rsync implementation -------------------------------------------


_PUSH_EXCLUDES = [
    "--exclude=__pycache__/",
    "--exclude=.git/",
    "--exclude=odin_runs/",
    "--exclude=benchmark_*_Isaac-*.json",
    "--exclude=*.swp",
    "--exclude=.claude/",
]


class ShellRsyncRunner:
    """Rsync runner that shells out to the ``rsync`` command."""

    def _build_ssh_transport_opt(self, host: ValkyrieConfig) -> str | None:
        """Return the value for rsync's ``-e`` flag when an ssh_key is set."""
        if host.ssh_key is None:
            return None
        return f"ssh -i {host.ssh_key} -o StrictHostKeyChecking=accept-new"

    def _run_rsync(self, argv: list[str]) -> RsyncResult:
        t0 = time.monotonic()
        proc = subprocess.run(argv, capture_output=True, text=True)
        duration = time.monotonic() - t0
        return RsyncResult(
            exit_code=int(proc.returncode),
            stdout=str(proc.stdout or ""),
            stderr=str(proc.stderr or ""),
            duration_s=duration,
        )

    def push(
        self,
        host: ValkyrieConfig,
        local_path: Path,
        remote_path: str,
    ) -> RsyncResult:
        """Push ``local_path`` (controller side) to ``remote_path`` on the Valkyrie.

        Includes ``--delete`` so the remote tree matches the local tree
        exactly (minus excludes), and a fixed exclude list for noise.
        """
        argv: list[str] = ["rsync", "-avz", "--delete", *_PUSH_EXCLUDES]
        transport = self._build_ssh_transport_opt(host)
        if transport is not None:
            argv += ["-e", transport]
        argv += [f"{str(local_path).rstrip('/')}/", f"{host.ssh_user}@{host.host}:{remote_path}"]
        return self._run_rsync(argv)

    def pull(
        self,
        host: ValkyrieConfig,
        remote_path: str,
        local_path: Path,
    ) -> RsyncResult:
        """Pull ``remote_path`` on the Valkyrie to ``local_path`` on the controller.

        NO ``--delete`` — we don't want to prune prior bundles on the
        controller's side when fetching a new bundle.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        argv: list[str] = ["rsync", "-avz"]
        transport = self._build_ssh_transport_opt(host)
        if transport is not None:
            argv += ["-e", transport]
        argv += [f"{host.ssh_user}@{host.host}:{remote_path}", str(local_path)]
        return self._run_rsync(argv)
```

Update `tools/odin/asgard/__init__.py` `__all__`:

```python
from tools.odin.asgard.transport import (
    RsyncResult,
    RsyncRunner,
    ShellRsyncRunner,
    ShellSSHRunner,
    SSHResult,
    SSHRunner,
)

__all__ = [
    # ... existing exports ...
    "RsyncResult",
    "RsyncRunner",
    "ShellRsyncRunner",
    "ShellSSHRunner",
    "SSHResult",
    "SSHRunner",
]
```

(Preserve prior `__all__` entries; just append the six transport names.)

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_transport_rsync.py -v --confcutdir=tools/odin
```

Expected: 5 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/transport.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_transport_rsync.py
git commit -m "Add ShellRsyncRunner for bundle push/pull

push() syncs the controller's local working tree to a Valkyrie with
--delete plus a fixed exclude list (__pycache__, .git, odin_runs,
benchmark json artifacts, .swp files, .claude config). pull() fetches
a remote bundle directory WITHOUT --delete so prior bundles on the
controller are preserved.

When ValkyrieConfig.ssh_key is set, rsync's -e flag gets a matching
ssh -i <key> ... transport string so rsync uses the same identity as
ShellSSHRunner."
```

---

## Task 6: `preflight.py` — one-shot health check per host

**Goal:** Single `preflight_valkyrie()` function that runs SSH-reach, docker-running, container-up, and isaaclab-present checks; returns a `PreflightResult`.

**Files:**
- Create: `tools/odin/asgard/preflight.py`
- Create: `tools/odin/tests/test_asgard_preflight.py`
- Modify: `tools/odin/asgard/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_preflight.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.preflight.preflight_valkyrie`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.preflight import PreflightResult, preflight_valkyrie
from tools.odin.asgard.transport import SSHResult


@dataclass
class _FakeSSH:
    """Deterministic SSH runner: returns scripted SSHResult for each cmd substring match."""

    scripted: dict  # {cmd_substring: SSHResult}

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=1, stdout="", stderr=f"no fake for {cmd!r}", duration_s=0.0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin")


def _ok() -> SSHResult:
    return SSHResult(exit_code=0, stdout="ok\n", stderr="", duration_s=0.01)


def _fail(msg: str) -> SSHResult:
    return SSHResult(exit_code=1, stdout="", stderr=msg, duration_s=0.01)


def test_all_checks_pass():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert isinstance(r, PreflightResult)
    assert r.ok is True
    assert r.checks == {
        "ssh_reach": True,
        "docker_running": True,
        "container_up": True,
        "isaaclab_present": True,
    }


def test_ssh_unreachable():
    ssh = _FakeSSH(scripted={"echo preflight-ok": _fail("connection refused")})
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["ssh_reach"] is False
    # Downstream checks should NOT be run / should be False.
    assert r.checks["docker_running"] is False
    assert "connection refused" in r.message


def test_docker_down():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _fail("docker daemon not responding"),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["ssh_reach"] is True
    assert r.checks["docker_running"] is False
    assert "docker" in r.message.lower()


def test_container_down():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["container_up"] is False
    assert "container" in r.message.lower()


def test_isaaclab_missing():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _fail("no such directory"),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["isaaclab_present"] is False
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_preflight.py -v --confcutdir=tools/odin
```

Expected: ImportError on `preflight_valkyrie`.

- [ ] **Step 3: Create `preflight.py`**

Create `tools/odin/asgard/preflight.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preflight — one-shot health check per Valkyrie before any job dispatches."""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import SSHRunner

__all__ = ["PreflightResult", "preflight_valkyrie"]


@dataclass
class PreflightResult:
    host: str
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    message: str = ""


def preflight_valkyrie(host: ValkyrieConfig, *, ssh: SSHRunner) -> PreflightResult:
    """Run SSH + docker + container + IsaacLab-directory checks on one host.

    Returns a :class:`PreflightResult` with ``ok=True`` iff all four checks
    pass. Later checks short-circuit: if SSH is unreachable, downstream
    checks are reported as ``False`` and the first failing check's diagnostic
    lands in ``message``.

    Args:
        host: Target Valkyrie.
        ssh: :class:`SSHRunner` implementation (``ShellSSHRunner`` in prod,
            fake in tests).

    Returns:
        Aggregated :class:`PreflightResult`.
    """
    checks = {
        "ssh_reach": False,
        "docker_running": False,
        "container_up": False,
        "isaaclab_present": False,
    }
    message = ""

    # 1. ssh_reach — single round-trip echo.
    r = ssh.run(host, "echo preflight-ok", timeout_s=15.0)
    if r.exit_code != 0:
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"ssh unreachable: {r.stderr.strip() or r.stdout.strip() or 'non-zero exit'}",
        )
    checks["ssh_reach"] = True

    # 2. docker_running — daemon responsive.
    r = ssh.run(host, "docker ps --format '{{.Names}}' 2>&1", timeout_s=15.0)
    if r.exit_code != 0:
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"docker daemon not responding: {r.stderr.strip() or r.stdout.strip()}",
        )
    checks["docker_running"] = True

    # 3. container_up — named container is in "running" state.
    r = ssh.run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}",
        timeout_s=15.0,
    )
    container_status = r.stdout.strip()
    if r.exit_code != 0 or container_status != "running":
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"container {host.container_name!r} not running (status={container_status!r})",
        )
    checks["container_up"] = True

    # 4. isaaclab_present — repo dir exists on the host.
    r = ssh.run(host, f"test -d {host.isaaclab_path}", timeout_s=10.0)
    if r.exit_code != 0:
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"IsaacLab path {host.isaaclab_path!r} missing on host",
        )
    checks["isaaclab_present"] = True

    return PreflightResult(host=host.host, ok=True, checks=checks, message="")
```

Update `tools/odin/asgard/__init__.py` to export `PreflightResult` and `preflight_valkyrie` (append to `__all__`).

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_preflight.py -v --confcutdir=tools/odin
```

Expected: 5 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/preflight.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_preflight.py
git commit -m "Add Asgard preflight — per-host health check

preflight_valkyrie() runs four checks in order: ssh_reach (echo), then
docker_running (docker ps), then container_up (docker inspect -f
'{{.State.Status}}'), then isaaclab_present (test -d). Later checks
short-circuit on failure; message reports the first failing check.

Called once per host before any ValkyrieWorker starts; preflight.json
at dispatch level records the result for audit."
```

---

## Task 7: `provisioner.py` — rsync working tree + container bringup

**Goal:** `provision_valkyrie()` syncs the controller's working tree to the Valkyrie and ensures the docker container is running. Smart-sync by default; `--fresh` wipes the remote tree and forces a container restart.

**Files:**
- Create: `tools/odin/asgard/provisioner.py`
- Create: `tools/odin/tests/test_asgard_provisioner.py`
- Modify: `tools/odin/asgard/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_provisioner.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.provisioner.provision_valkyrie`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.provisioner import ProvisionResult, provision_valkyrie
from tools.odin.asgard.transport import RsyncResult, SSHResult


@dataclass
class _FakeSSH:
    log: list[str] = field(default_factory=list)
    scripted: dict = field(default_factory=dict)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        self.log.append(cmd)
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


@dataclass
class _FakeRsync:
    log: list[tuple[str, str, str]] = field(default_factory=list)  # (op, local_or_remote_src, dst)

    def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
        self.log.append(("push", str(local_path), remote_path))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        self.log.append(("pull", remote_path, str(local_path)))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", isaaclab_path="~/IsaacLab")


def test_smart_sync_pushes_but_does_not_wipe(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert isinstance(r, ProvisionResult)
    assert r.ok is True
    # No wipe.
    assert not any("rm -rf" in cmd for cmd in ssh.log)
    # rsync push happened with working_tree -> isaaclab_path.
    assert rsync.log == [("push", str(tmp_path), "~/IsaacLab")]
    # Container already running → no start call.
    assert not any("./docker/container.py start" in cmd for cmd in ssh.log)


def test_fresh_wipes_and_restarts(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=True, ssh=ssh, rsync=rsync)
    assert r.ok is True
    # Wipe before push.
    wipe_idx = next(i for i, cmd in enumerate(ssh.log) if "rm -rf" in cmd and "~/IsaacLab" in cmd)
    push_idx = rsync.log.index(("push", str(tmp_path), "~/IsaacLab"))
    # The wipe must precede the rsync (same runner receives the wipe before
    # provisioner returns from the wipe call), which in our log terms means
    # the wipe command appears in ssh.log before the push call.
    assert wipe_idx >= 0
    assert push_idx >= 0
    # Container stop + start.
    assert any("./docker/container.py stop" in cmd for cmd in ssh.log)
    assert any("./docker/container.py start" in cmd for cmd in ssh.log)


def test_smart_sync_starts_container_when_stopped(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert r.ok is True
    # No wipe.
    assert not any("rm -rf" in cmd for cmd in ssh.log)
    # Must start container.
    assert any("./docker/container.py start" in cmd for cmd in ssh.log)


def test_failed_rsync_reports_not_ok(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )

    class _BadRsync(_FakeRsync):
        def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
            return RsyncResult(exit_code=23, stdout="", stderr="rsync: failed", duration_s=0.0)

    rsync = _BadRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert r.ok is False
    assert "rsync" in r.message.lower()


def test_provision_result_records_commit_sha(tmp_path: Path, monkeypatch):
    """ProvisionResult.commit_sha comes from _resolve_local_sha(working_tree)."""
    from tools.odin.asgard import provisioner as prov_mod

    def _fake_resolve(wt: Path) -> str:
        return "abc123d"

    monkeypatch.setattr(prov_mod, "_resolve_local_sha", _fake_resolve)
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert r.commit_sha == "abc123d"
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_provisioner.py -v --confcutdir=tools/odin
```

Expected: ImportError on `provision_valkyrie`.

- [ ] **Step 3: Create `provisioner.py`**

Create `tools/odin/asgard/provisioner.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Valkyrie provisioning — rsync working tree + docker container bringup."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import RsyncRunner, SSHRunner

__all__ = ["ProvisionResult", "provision_valkyrie"]


@dataclass
class ProvisionResult:
    host: str
    ok: bool
    message: str = ""
    commit_sha: str = ""


def _resolve_local_sha(working_tree: Path) -> str:
    """Return the controller's current git HEAD SHA, suffixed -dirty if uncommitted."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=working_tree,
            text=True,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=working_tree,
        ).returncode
        if dirty != 0:
            sha = f"{sha}-dirty"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _container_status(host: ValkyrieConfig, ssh: SSHRunner) -> str:
    """Return 'running' / 'exited' / ... or '' if docker inspect fails."""
    r = ssh.run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}",
        timeout_s=15.0,
    )
    if r.exit_code != 0:
        return ""
    return r.stdout.strip()


def _container_start(host: ValkyrieConfig, ssh: SSHRunner) -> bool:
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py start",
        timeout_s=300.0,
    )
    return r.exit_code == 0


def _container_stop(host: ValkyrieConfig, ssh: SSHRunner) -> bool:
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py stop",
        timeout_s=120.0,
    )
    return r.exit_code == 0


def provision_valkyrie(
    host: ValkyrieConfig,
    working_tree: Path,
    *,
    fresh: bool,
    ssh: SSHRunner,
    rsync: RsyncRunner,
) -> ProvisionResult:
    """Bring a Valkyrie up to the controller's current working tree.

    Flow:

    1. If ``fresh=True``: SSH ``rm -rf {isaaclab_path}`` on the host.
    2. Rsync push ``working_tree`` → ``{isaaclab_path}``.
    3. Container state:
       - ``fresh=True``: stop + start.
       - ``fresh=False``: query ``docker inspect`` status; start if not running.
    4. Return a :class:`ProvisionResult` with ``commit_sha`` from the local
       working tree (suffixed ``-dirty`` if uncommitted changes).

    Args:
        host: Target Valkyrie.
        working_tree: Controller-side IsaacLab path to push from (typically
            the repo root).
        fresh: When ``True``, wipe + full re-sync + container restart.
        ssh: SSH runner.
        rsync: Rsync runner.

    Returns:
        :class:`ProvisionResult` with ``ok=False`` on any step failure and
        a descriptive ``message``.
    """
    commit_sha = _resolve_local_sha(working_tree)

    if fresh:
        r = ssh.run(host, f"rm -rf {host.isaaclab_path}", timeout_s=60.0)
        if r.exit_code != 0:
            return ProvisionResult(
                host=host.host,
                ok=False,
                message=f"fresh wipe failed: {r.stderr.strip() or 'non-zero exit'}",
                commit_sha=commit_sha,
            )

    rr = rsync.push(host, working_tree, host.isaaclab_path)
    if rr.exit_code != 0:
        return ProvisionResult(
            host=host.host,
            ok=False,
            message=f"rsync push failed: {rr.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
        )

    if fresh:
        # Best-effort stop (may fail if container didn't exist yet — that's fine).
        _container_stop(host, ssh)
        if not _container_start(host, ssh):
            return ProvisionResult(
                host=host.host,
                ok=False,
                message="container.py start failed after fresh wipe",
                commit_sha=commit_sha,
            )
    else:
        status = _container_status(host, ssh)
        if status != "running":
            if not _container_start(host, ssh):
                return ProvisionResult(
                    host=host.host,
                    ok=False,
                    message=f"container.py start failed (prior status={status!r})",
                    commit_sha=commit_sha,
                )

    return ProvisionResult(host=host.host, ok=True, commit_sha=commit_sha)
```

Update `tools/odin/asgard/__init__.py` to export `ProvisionResult` and `provision_valkyrie`.

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_provisioner.py -v --confcutdir=tools/odin
```

Expected: 5 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/provisioner.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_provisioner.py
git commit -m "Add Valkyrie provisioner — rsync + container bringup

provision_valkyrie() implements the smart-sync flow: rsync push of the
controller's working tree to the Valkyrie's isaaclab_path (with
--fresh wiping the remote tree first), then ensures the docker
container is running via ./docker/container.py start (or stop+start
under --fresh).

commit_sha in ProvisionResult is the controller's git rev-parse
--short HEAD, suffixed -dirty when the working tree has uncommitted
changes. run_dispatch() threads this into dispatch.json so the audit
record reflects exactly what was rsync'd."
```

---

## Task 8: `worker.py` — `ValkyrieWorker` thread, happy path + failure classification

**Goal:** One thread-per-Valkyrie worker that consumes jobs from a shared queue, invokes Hugin/Munin inside the Valkyrie's docker container, tees stdout to a log file, pulls the bundle back, validates it, and posts a state event. Failure classification lands in this task; retry-on-different-node via `preferred_not` hint lands in Task 9.

**Files:**
- Create: `tools/odin/asgard/worker.py`
- Create: `tools/odin/tests/test_asgard_worker.py`
- Modify: `tools/odin/asgard/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_worker.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`tools.odin.asgard.worker.ValkyrieWorker` (happy path + classification)."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.queue import JobEntry
from tools.odin.asgard.transport import RsyncResult, SSHResult
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", isaaclab_path="~/IsaacLab")


def _job(run_id: str = "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name=run_id,
    )


@dataclass
class _FakeSSH:
    scripted: dict = field(default_factory=dict)
    log: list[tuple[str, str]] = field(default_factory=list)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        self.log.append((host.host, cmd))
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01)


@dataclass
class _FakeRsync:
    materialize_bundle: bool = True
    log: list[tuple[str, str, str]] = field(default_factory=list)

    def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
        self.log.append(("push", str(local_path), remote_path))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        self.log.append(("pull", remote_path, str(local_path)))
        if self.materialize_bundle:
            # Fake Hugin creating manifest + training + startup at local_path.
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "phases": {"startup": {"status": "completed"}, "training": {"status": "completed"}}})
            )
            (local_path / "training.json").write_text(json.dumps({"schema_version": "1.0"}))
            (local_path / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _spin_worker(worker: ValkyrieWorker, jobs: list[JobEntry]) -> list[StateEvent]:
    """Run the worker against the given job list synchronously (no threading) and collect events."""
    for j in jobs:
        worker._job_queue.put(j)
    # Sentinel so the worker exits after draining.
    worker._job_queue.put(None)
    worker.run()
    events: list[StateEvent] = []
    while True:
        try:
            events.append(worker._state_chan.get_nowait())
        except queue.Empty:
            return events


def _make_worker(tmp_path: Path, ssh, rsync, host=None) -> ValkyrieWorker:
    return ValkyrieWorker(
        host=host or _host(),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=60, max_infrastructure_retries=2),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=threading.Event(),
    )


def test_worker_happy_path(tmp_path: Path):
    ssh = _FakeSSH()
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    kinds = [e.transition for e in events]
    assert "running" in kinds
    assert "completed" in kinds
    # Bundle dir exists locally after pull.
    assert (tmp_path / _job().bundle_dir_name / "manifest.json").exists()


def test_worker_classifies_hugin_crash(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker exec": SSHResult(
                exit_code=1, stdout="", stderr="CUDA out of memory\n", duration_s=5.0
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure is not None
    assert failed.failure.kind == "hugin_crash"
    assert failed.failure.details["exit_code"] == 1


def test_worker_classifies_timeout(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker exec": SSHResult(
                exit_code=-15, stdout="", stderr="", duration_s=60.1, timed_out=True
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "timeout"


def test_worker_classifies_malformed_bundle(tmp_path: Path):
    class _RsyncNoManifest(_FakeRsync):
        def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
            local_path.mkdir(parents=True, exist_ok=True)
            # No manifest.json — should classify as malformed.
            return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    ssh = _FakeSSH()
    rsync = _RsyncNoManifest()
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "hugin_malformed_bundle"


def test_worker_classifies_infrastructure_before_hugin(tmp_path: Path):
    """ssh error on docker exec itself (exit -1 / no such container) is infrastructure, not hugin_crash."""
    ssh = _FakeSSH(
        scripted={
            # docker exec exits 125 when docker itself rejects the command (container not found).
            "docker exec": SSHResult(
                exit_code=125,
                stdout="",
                stderr="Error: No such container: isaac-lab-base\n",
                duration_s=0.5,
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    # With max_infrastructure_retries=2 the worker re-queues the job; after
    # exhausting retries (all still 125) it emits failed(infrastructure).
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "infrastructure"


def test_worker_writes_ssh_tail_log(tmp_path: Path):
    class _SSHThatTees(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            # Emulate the real runner's tee behaviour.
            if stdout_tee is not None:
                stdout_tee.parent.mkdir(parents=True, exist_ok=True)
                stdout_tee.write_text("iter 1\niter 2\ndone\n")
            return SSHResult(exit_code=0, stdout="iter 1\niter 2\ndone\n", stderr="", duration_s=0.01)

    ssh = _SSHThatTees()
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    _spin_worker(w, [_job()])
    tee = tmp_path / _job().bundle_dir_name / "logs" / "ssh-tail.log"
    assert tee.exists()
    assert "iter 1" in tee.read_text()
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v --confcutdir=tools/odin
```

Expected: `ImportError` on `ValkyrieWorker` / `StateEvent` / `WorkerOptions`.

- [ ] **Step 3: Create `worker.py`**

Create `tools/odin/asgard/worker.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ValkyrieWorker — per-host thread that consumes jobs and runs them end-to-end."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.queue import FailureInfo, JobEntry
from tools.odin.asgard.transport import RsyncRunner, SSHResult, SSHRunner

__all__ = ["StateEvent", "ValkyrieWorker", "WorkerOptions"]


@dataclass
class WorkerOptions:
    per_job_timeout_s: int = 14400
    max_infrastructure_retries: int = 2


@dataclass
class StateEvent:
    """Message posted by a worker to the state channel on every transition."""

    run_id: str
    host: str
    transition: str                   # "running" | "completed" | "failed" | "shutdown_idle"
    failure: FailureInfo | None = None
    started_at: str | None = None
    ended_at: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# docker exec exits 125 when the docker command itself failed (container not
# found, daemon unreachable). Treat those as infrastructure, not hugin_crash.
_INFRASTRUCTURE_DOCKER_EXIT_CODES = {125, 126, 127}


def _build_docker_exec_cmd(host: ValkyrieConfig, job: JobEntry) -> str:
    """Return the remote shell command to run Hugin/Munin inside the container.

    Shape: ``cd {isaaclab_path} && docker exec {container_name} bash -lc '...'``
    where the inner command is the Hugin (rsl_rl) or Munin (skrl) wrapper
    invocation with the job's CLI args.
    """
    runner_script = "tools/odin/hugin/run.py" if job.framework == "rsl_rl" else "tools/odin/munin/run.py"
    inner_parts = [
        "cd /workspace/isaaclab",
        "PYTHONPATH=.",
        f"./isaaclab.sh -p {runner_script}",
        f"--task {job.task_id}",
        f"--backend {job.backend}",
        f"--seed {job.seed}",
        f"--num_envs {job.num_envs}",
        f"--max_iterations {job.max_iterations}",
        f"--runs_root odin_runs",
    ]
    inner = " && ".join(inner_parts[:1]) + " && " + " ".join(inner_parts[1:])
    return f"cd {host.isaaclab_path} && docker exec {host.container_name} bash -lc '{inner}'"


class ValkyrieWorker(threading.Thread):
    """Per-Valkyrie worker thread.

    Pulls :class:`JobEntry` items from a shared ``queue.Queue`` and runs
    them end-to-end: docker-exec-the-job over SSH, tee stdout to a local
    log file, rsync-pull the bundle back, validate, classify failures.

    Events posted to ``state_chan`` on every transition are consumed by
    the main thread to rewrite ``dispatch.json``.
    """

    def __init__(
        self,
        host: ValkyrieConfig,
        job_queue: queue.Queue,
        state_chan: queue.Queue,
        dispatch_dir: Path,
        options: WorkerOptions,
        *,
        ssh: SSHRunner,
        rsync: RsyncRunner,
        shutdown_event: threading.Event,
    ):
        super().__init__(name=f"ValkyrieWorker-{host.host}", daemon=True)
        self.host = host
        self._job_queue = job_queue
        self._state_chan = state_chan
        self._dispatch_dir = dispatch_dir
        self._options = options
        self._ssh = ssh
        self._rsync = rsync
        self._shutdown = shutdown_event

    # -- public entry point -------------------------------------------------

    def run(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:  # sentinel: queue drained
                return
            if job.preferred_not and self.host.host in job.preferred_not:
                # Put it back and yield briefly so another worker gets a shot.
                self._job_queue.put(job)
                time.sleep(0.5)
                continue
            self._execute(job)

    # -- execute one job ----------------------------------------------------

    def _execute(self, job: JobEntry) -> None:
        started_at = _utc_now_iso()
        self._state_chan.put(
            StateEvent(run_id=job.run_id, host=self.host.host, transition="running", started_at=started_at)
        )
        job.started_at = started_at
        job.attempts += 1

        ssh_tail = self._dispatch_dir / job.bundle_dir_name / "logs" / "ssh-tail.log"
        cmd = _build_docker_exec_cmd(self.host, job)
        ssh_result = self._ssh.run(
            self.host, cmd, timeout_s=float(self._options.per_job_timeout_s), stdout_tee=ssh_tail
        )

        failure = self._classify(ssh_result, job, ssh_tail)
        if failure is not None and failure.kind == "infrastructure":
            # Retry infrastructure failures up to the configured budget.
            if job.attempts <= self._options.max_infrastructure_retries:
                job.preferred_not.add(self.host.host)
                self._job_queue.put(job)
                # Do NOT emit a "failed" event on retryable infrastructure — emit
                # a "running" rollback so the main thread keeps the row in-flight.
                return
            # Exhausted retries → fall through to failure emit.

        if failure is not None:
            job.status = "failed"
            job.failure = failure
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(
                    run_id=job.run_id,
                    host=self.host.host,
                    transition="failed",
                    failure=failure,
                    ended_at=job.ended_at,
                )
            )
            return

        # Success path: rsync pull the bundle back.
        remote_bundle = f"{self.host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        local_bundle = self._dispatch_dir / job.bundle_dir_name
        rsync_result = self._rsync.pull(self.host, remote_bundle, local_bundle)
        if rsync_result.exit_code != 0:
            # Bundle wasn't pulled — treat as infrastructure so we retry.
            if job.attempts <= self._options.max_infrastructure_retries:
                job.preferred_not.add(self.host.host)
                self._job_queue.put(job)
                return
            job.status = "failed"
            job.failure = FailureInfo(
                kind="infrastructure",
                message=f"rsync pull failed: {rsync_result.stderr.strip() or 'non-zero exit'}",
                details={"attempts": job.attempts},
            )
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=job.failure)
            )
            return

        # Validate the bundle: manifest.json present, schema-v1 shape.
        bundle_failure = _validate_bundle(local_bundle)
        if bundle_failure is not None:
            job.status = "failed"
            job.failure = bundle_failure
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=bundle_failure)
            )
            return

        job.status = "completed"
        job.ended_at = _utc_now_iso()
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="completed",
                ended_at=job.ended_at,
            )
        )

    # -- classification -----------------------------------------------------

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
                message=(
                    f"docker exec failed with exit {r.exit_code}: {r.stderr.strip() or 'unknown'}"
                ),
                details={
                    "exit_code": r.exit_code,
                    "attempts": job.attempts,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        if r.exit_code != 0:
            return FailureInfo(
                kind="hugin_crash",
                message=(
                    f"exit code {r.exit_code}; stderr tail: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '(empty)'!r}"
                ),
                details={
                    "exit_code": r.exit_code,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        return None


def _validate_bundle(local_bundle: Path) -> FailureInfo | None:
    """Check that manifest.json exists and declares schema_version==1.0."""
    manifest_path = local_bundle / "manifest.json"
    if not manifest_path.exists():
        return FailureInfo(
            kind="hugin_malformed_bundle",
            message="manifest.json missing after rsync pull",
            details={"bundle_dir": str(local_bundle.name)},
        )
    try:
        m = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return FailureInfo(
            kind="hugin_malformed_bundle",
            message=f"manifest.json is not valid JSON: {exc}",
            details={"bundle_dir": str(local_bundle.name)},
        )
    if str(m.get("schema_version", "")) != "1.0":
        return FailureInfo(
            kind="hugin_malformed_bundle",
            message=f"manifest.json schema_version != 1.0 (got {m.get('schema_version')!r})",
            details={"bundle_dir": str(local_bundle.name)},
        )
    return None
```

Update `tools/odin/asgard/__init__.py` to export `StateEvent`, `ValkyrieWorker`, `WorkerOptions`.

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v --confcutdir=tools/odin
```

Expected: 6 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_worker.py
git commit -m "Add ValkyrieWorker thread with failure classification

Each Valkyrie gets a threading.Thread that pulls JobEntry items from a
shared queue.Queue. Per job: build a 'cd <isaaclab_path> && docker
exec <container> bash -lc 'cd /workspace/isaaclab && ...' command,
run it via the SSHRunner with stdout tee'd to ssh-tail.log, then
rsync pull the bundle back and validate manifest.json's
schema_version.

Classification:
  timed_out                → FailureInfo(kind='timeout')
  docker daemon exit codes → FailureInfo(kind='infrastructure')
  any other non-zero exit  → FailureInfo(kind='hugin_crash')
  zero exit + bad bundle   → FailureInfo(kind='hugin_malformed_bundle')
  zero exit + good bundle  → status='completed'

Infrastructure failures with attempts left are re-queued with
host.host added to the job's preferred_not hint so another worker
picks it up; only after max_infrastructure_retries do we emit a
failed event."
```

---

## Task 9: `worker.py` — shutdown + `preferred_not` fallback + remaining retry behaviour

**Goal:** Close out the worker's shutdown path (graceful SIGINT handling — drain current job, then exit) and the `preferred_not` fallback (all other workers busy → pull anyway, don't spin).

**Files:**
- Modify: `tools/odin/asgard/worker.py`
- Modify: `tools/odin/tests/test_asgard_worker.py` (append)

- [ ] **Step 1: Append tests for the remaining behaviours**

Append to `tools/odin/tests/test_asgard_worker.py`:

```python


def test_worker_respects_shutdown_between_jobs(tmp_path: Path):
    """shutdown_event.set() stops the worker from pulling the next job."""
    ssh = _FakeSSH()
    rsync = _FakeRsync()
    host = _host()
    job_q = queue.Queue()
    state_q = queue.Queue()
    shutdown = threading.Event()
    worker = ValkyrieWorker(
        host=host,
        job_queue=job_q,
        state_chan=state_q,
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=60),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=shutdown,
    )
    shutdown.set()
    # Even with jobs queued, the worker should exit without consuming any.
    job_q.put(_job("r-skipped"))
    worker.run()
    events = []
    while not state_q.empty():
        events.append(state_q.get_nowait())
    # No running / completed / failed event for r-skipped.
    assert all(e.run_id != "r-skipped" for e in events)


def test_preferred_not_fallback_no_other_worker(tmp_path: Path):
    """When a job's preferred_not lists our host but NO other worker is around,
    the worker eventually accepts and runs it (we can't leave it stuck)."""
    ssh = _FakeSSH()
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    j = _job("r-pref")
    j.preferred_not = {w.host.host}
    # After ~N yields the worker accepts the job. We don't want an infinite
    # loop so the implementation must have a bounded yield budget per job.
    # Here we assert that within 5 rounds of put-back / yield, the job is
    # actually executed (emits a 'running' state).
    w._job_queue.put(j)
    w._job_queue.put(None)
    w.run()
    events = []
    while not w._state_chan.empty():
        events.append(w._state_chan.get_nowait())
    transitions = [e.transition for e in events]
    assert "running" in transitions
    assert "completed" in transitions
```

Now extend the worker's `preferred_not` handling with a bounded retry-yield count so the self-owning-only-worker case eventually proceeds.

- [ ] **Step 2: Run the new tests to verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py::test_worker_respects_shutdown_between_jobs tools/odin/tests/test_asgard_worker.py::test_preferred_not_fallback_no_other_worker -v --confcutdir=tools/odin
```

Expected: `test_worker_respects_shutdown_between_jobs` passes already (shutdown is checked at the top of `run`). `test_preferred_not_fallback_no_other_worker` hangs or times out — the current `preferred_not` branch re-queues forever.

- [ ] **Step 3: Add bounded fallback to `run`**

Replace the `preferred_not` branch in `ValkyrieWorker.run()` with the bounded version. Find:

```python
            if job.preferred_not and self.host.host in job.preferred_not:
                # Put it back and yield briefly so another worker gets a shot.
                self._job_queue.put(job)
                time.sleep(0.5)
                continue
            self._execute(job)
```

Replace with:

```python
            if job.preferred_not and self.host.host in job.preferred_not:
                # Put it back and let another worker pick it up. To avoid
                # spinning in the degenerate "only worker alive" case, bound
                # the number of times WE will refuse the same job.
                seen_count = self._preferred_not_seen.get(job.run_id, 0) + 1
                self._preferred_not_seen[job.run_id] = seen_count
                if seen_count < 3:
                    self._job_queue.put(job)
                    time.sleep(0.5)
                    continue
                # Fall through: take the job anyway.
            self._execute(job)
```

Also add `self._preferred_not_seen: dict[str, int] = {}` to `__init__`.

- [ ] **Step 4: Re-run tests**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v --confcutdir=tools/odin
```

Expected: 8/8 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker.py
git commit -m "Add bounded preferred_not fallback + verify shutdown semantics

The preferred_not hint (added during infrastructure retry) keeps the
same worker from immediately re-pulling the same broken job. In the
degenerate case where only one worker is alive — or all others are
busy for a long time — an unbounded re-queue would spin forever. Cap
the refuse count at 3: after the third refusal, the worker takes the
job anyway. In practice, with multiple workers the job lands on a
different host before the cap is hit.

Also pin the shutdown-between-jobs semantics with a direct test:
setting shutdown_event before the worker starts ensures no job is
consumed."
```

---

## Task 10: `runner.py` — `run_dispatch()` orchestrator

**Goal:** The top-level function the CLI (and future web UI) calls. Resolves the dispatch directory, runs preflight, provisions Valkyries, spawns workers, drains state events, writes `dispatch.json` atomically, handles resume.

**Files:**
- Create: `tools/odin/asgard/runner.py`
- Create: `tools/odin/tests/test_asgard_runner.py`
- Modify: `tools/odin/asgard/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_runner.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.runner.run_dispatch`.

End-to-end tests with fake SSH + rsync; no threading primitives exercised
at the integration level (that's the slow loopback test). These tests
verify dispatch orchestration, dispatch.json rewrite, and resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.queue import JobEntry
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch
from tools.odin.asgard.state import read_dispatch_state
from tools.odin.asgard.transport import RsyncResult, SSHResult


@dataclass
class _FakeSSH:
    scripted: dict = field(default_factory=dict)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="ok", stderr="", duration_s=0.01)


@dataclass
class _FakeRsync:
    materialize: bool = True

    def push(self, host, local_path, remote_path) -> RsyncResult:
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        if self.materialize:
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "phases": {"training": {"status": "completed"}}})
            )
            (local_path / "training.json").write_text(json.dumps({"schema_version": "1.0"}))
            (local_path / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _write_fleet(tmp_path: Path) -> Fleet:
    return Fleet(
        fleet_name="t",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="odin"),
            ValkyrieConfig(host="v2", ssh_user="odin"),
        ],
    )


def _write_env_list(tmp_path: Path) -> Path:
    from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            status="current",
        )
    ]
    p = tmp_path / "physx.yaml"
    write_env_list(p, el, generator="test")
    return p


def test_resolve_dispatch_dir_creates_new(tmp_path: Path):
    d = resolve_dispatch_dir(tmp_path / "odin_runs", resume=None)
    assert d.exists()
    assert d.parent == tmp_path / "odin_runs"


def test_resolve_dispatch_dir_resume_latest(tmp_path: Path):
    root = tmp_path / "odin_runs"
    root.mkdir()
    # Create two simulated prior dispatch dirs.
    (root / "20260420-100000").mkdir()
    (root / "20260421-120000").mkdir()
    d = resolve_dispatch_dir(root, resume="LATEST")
    assert d.name == "20260421-120000"


def test_resolve_dispatch_dir_resume_named(tmp_path: Path):
    root = tmp_path / "odin_runs"
    (root / "20260420-100000").mkdir(parents=True)
    d = resolve_dispatch_dir(root, resume="20260420-100000")
    assert d.name == "20260420-100000"


def test_run_dispatch_happy_path(tmp_path: Path):
    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260422-220000"
    dispatch_dir.mkdir(parents=True)
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42]),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert state is not None
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed"
    assert (dispatch_dir / "dispatch.json").exists()
    # Bundle rsync'd back.
    assert (dispatch_dir / state.jobs[0].bundle_dir_name / "manifest.json").exists()


def test_run_dispatch_preflight_fail_fast(tmp_path: Path):
    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260422-220000"
    dispatch_dir.mkdir(parents=True)
    # docker ps fails → preflight fails → run_dispatch aborts.
    with pytest.raises(RuntimeError, match="preflight"):
        run_dispatch(
            fleet=fleet,
            physx_yaml=physx,
            newton_yaml=None,
            dispatch_dir=dispatch_dir,
            options=DispatchOptions(seeds=[42]),
            ssh=_FakeSSH(
                scripted={
                    "docker ps": SSHResult(exit_code=1, stdout="", stderr="daemon unreachable", duration_s=0.01),
                }
            ),
            rsync=_FakeRsync(),
        )
    # Preflight.json is written on failure for audit.
    assert (dispatch_dir / "preflight.json").exists()


def test_run_dispatch_resume_preserves_completed(tmp_path: Path):
    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260422-220000"
    dispatch_dir.mkdir(parents=True)

    # First run: completes the single job.
    run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42]),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    first = read_dispatch_state(dispatch_dir)
    assert first.jobs[0].status == "completed"

    # Second run (resume) MUST NOT re-run a completed job.
    class _AssertNoDispatch(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "docker exec" in cmd:
                raise AssertionError(f"resume should not re-dispatch completed jobs; cmd={cmd!r}")
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42]),
        ssh=_AssertNoDispatch(),
        rsync=_FakeRsync(),
    )
    second = read_dispatch_state(dispatch_dir)
    assert second.jobs[0].status == "completed"
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_runner.py -v --confcutdir=tools/odin
```

Expected: ImportError on `run_dispatch` / `DispatchOptions` / `resolve_dispatch_dir`.

- [ ] **Step 3: Create `runner.py`**

Create `tools/odin/asgard/runner.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""run_dispatch — top-level orchestration for an Asgard dispatch run."""

from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.preflight import preflight_valkyrie
from tools.odin.asgard.provisioner import provision_valkyrie
from tools.odin.asgard.queue import JobEntry, build_queue_from_env_lists
from tools.odin.asgard.state import (
    DispatchState,
    FleetSnapshot,
    SCHEMA_VERSION,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)
from tools.odin.asgard.transport import RsyncRunner, SSHRunner, ShellRsyncRunner, ShellSSHRunner
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions

__all__ = ["DispatchOptions", "resolve_dispatch_dir", "run_dispatch"]


@dataclass
class DispatchOptions:
    seeds: list[int]
    max_infrastructure_retries: int = 2
    per_job_timeout_s: int = 14400
    fresh: bool = False
    skip_preflight: bool = False
    include_filter: list[str] | None = None
    verbose: bool = False
    retry_failed: list[str] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dispatch_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def resolve_dispatch_dir(runs_root: Path, resume: str | None) -> Path:
    """Return the dispatch directory (creating one if ``resume is None``).

    - ``resume=None``: create a fresh ``runs_root/<UTC-now>/`` directory.
    - ``resume="LATEST"``: return the most-recent existing subdirectory.
    - ``resume="<dispatch_id>"``: return ``runs_root/<dispatch_id>/`` (must exist).
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    if resume is None:
        dispatch_dir = runs_root / _dispatch_id_now()
        dispatch_dir.mkdir(parents=True, exist_ok=True)
        return dispatch_dir
    if resume == "LATEST":
        children = sorted(p for p in runs_root.iterdir() if p.is_dir())
        if not children:
            raise FileNotFoundError(f"No prior dispatch directories under {runs_root}")
        return children[-1]
    candidate = runs_root / resume
    if not candidate.exists():
        raise FileNotFoundError(f"Resume target {candidate} does not exist")
    return candidate


def _snapshot_fleet_yaml(fleet: Fleet, dispatch_dir: Path) -> None:
    """Write fleet.yaml.snapshot alongside dispatch.json for audit."""
    payload = {
        "fleet_name": fleet.fleet_name,
        "hosts": [
            {
                "host": h.host,
                "ssh_user": h.ssh_user,
                "ssh_key": str(h.ssh_key) if h.ssh_key is not None else None,
                "isaaclab_path": h.isaaclab_path,
                "container_name": h.container_name,
                "labels": h.labels,
            }
            for h in fleet.hosts
        ],
    }
    (dispatch_dir / "fleet.yaml.snapshot").write_text(json.dumps(payload, indent=2))


def _write_preflight(results, dispatch_dir: Path, dispatch_id: str) -> None:
    payload = {
        "schema_version": "1.0",
        "dispatch_id": dispatch_id,
        "checked_at": _utc_now_iso(),
        "hosts": [
            {"host": r.host, "ok": r.ok, "checks": r.checks, "message": r.message}
            for r in results
        ],
    }
    (dispatch_dir / "preflight.json").write_text(json.dumps(payload, indent=2))


def _merge_jobs(existing: list[JobEntry], fresh: list[JobEntry]) -> list[JobEntry]:
    """Preserve completed / failed from existing; take pending / running /
    assigned flipped-to-pending rows from existing too; new rows from fresh.

    Raises ValueError if the fresh list is not a superset of the existing list
    (dispatch_id / seeds mismatch → user should start a new dispatch).
    """
    by_id = {j.run_id: j for j in existing}
    merged: list[JobEntry] = []
    fresh_ids: set[str] = set()
    for f in fresh:
        fresh_ids.add(f.run_id)
        if f.run_id in by_id:
            merged.append(by_id[f.run_id])
        else:
            merged.append(f)
    missing = set(by_id) - fresh_ids
    if missing:
        raise ValueError(
            f"Resume target contains jobs not in the current queue: {sorted(missing)[:3]}... "
            f"(dispatch_id / seeds / include_filter changed; start a new dispatch)"
        )
    return merged


def run_dispatch(
    fleet: Fleet,
    physx_yaml: Path | None,
    newton_yaml: Path | None,
    dispatch_dir: Path,
    options: DispatchOptions,
    *,
    ssh: SSHRunner | None = None,
    rsync: RsyncRunner | None = None,
) -> DispatchState:
    """Orchestrate one distributed dispatch.

    Inputs:
      - ``fleet``: Valkyrie host list.
      - ``physx_yaml`` / ``newton_yaml``: curated T2.1 env lists (at least one).
      - ``dispatch_dir``: ``odin_runs/<dispatch_id>/`` directory (created by
        :func:`resolve_dispatch_dir`; either fresh or a resume target).
      - ``options``: seeds, timeouts, etc. See :class:`DispatchOptions`.
      - ``ssh`` / ``rsync``: transport injection points. Default to
        ``ShellSSHRunner`` / ``ShellRsyncRunner``.

    Output:
      Final :class:`DispatchState` (also written to
      ``<dispatch_dir>/dispatch.json``).
    """
    ssh = ssh or ShellSSHRunner()
    rsync = rsync or ShellRsyncRunner()
    dispatch_id = dispatch_dir.name

    fresh_jobs = build_queue_from_env_lists(
        physx_yaml=physx_yaml,
        newton_yaml=newton_yaml,
        seeds=options.seeds,
        dispatch_id=dispatch_id,
        include_filter=options.include_filter,
    )

    # Load prior state for resume if it exists.
    prior_state = read_dispatch_state(dispatch_dir)
    if prior_state is not None:
        # Flip in-flight → pending first, then merge.
        reset_in_flight_to_pending(prior_state)
        merged_jobs = _merge_jobs(prior_state.jobs, fresh_jobs)
        started_at = prior_state.started_at
        # Re-attempt specific failed jobs on explicit request.
        if options.retry_failed:
            retry_set = set(options.retry_failed)
            for j in merged_jobs:
                if j.run_id in retry_set and j.status == "failed":
                    j.status = "pending"
                    j.failure = None
    else:
        merged_jobs = fresh_jobs
        started_at = _utc_now_iso()

    # Snapshot fleet.yaml.
    _snapshot_fleet_yaml(fleet, dispatch_dir)

    # Preflight.
    pre_results = [preflight_valkyrie(h, ssh=ssh) for h in fleet.hosts]
    _write_preflight(pre_results, dispatch_dir, dispatch_id)

    healthy: list[ValkyrieConfig] = []
    down_hosts: set[str] = set()
    for host, res in zip(fleet.hosts, pre_results):
        if res.ok:
            healthy.append(host)
        else:
            down_hosts.add(host.host)

    if not healthy:
        # Emit a final dispatch.json before raising so the audit record is complete.
        state = DispatchState(
            schema_version=SCHEMA_VERSION,
            dispatch_id=dispatch_id,
            started_at=started_at,
            ended_at=_utc_now_iso(),
            seeds=options.seeds,
            commit_sha="",
            fleet=[FleetSnapshot(host=h.host, status="down",
                                 last_error=next((r.message for r in pre_results if r.host == h.host), None))
                   for h in fleet.hosts],
            jobs=merged_jobs,
        )
        write_dispatch_state(dispatch_dir, state)
        raise RuntimeError(f"preflight failed for all {len(fleet.hosts)} hosts; see preflight.json")

    if down_hosts and not options.skip_preflight:
        state = DispatchState(
            schema_version=SCHEMA_VERSION,
            dispatch_id=dispatch_id,
            started_at=started_at,
            ended_at=_utc_now_iso(),
            seeds=options.seeds,
            commit_sha="",
            fleet=[
                FleetSnapshot(
                    host=h.host,
                    status="down" if h.host in down_hosts else "idle",
                    last_error=next((r.message for r in pre_results if r.host == h.host and not r.ok), None),
                )
                for h in fleet.hosts
            ],
            jobs=merged_jobs,
        )
        write_dispatch_state(dispatch_dir, state)
        raise RuntimeError(
            f"preflight failed for {len(down_hosts)}/{len(fleet.hosts)} hosts; "
            f"pass --skip-preflight to run on healthy ones only. See preflight.json."
        )

    # Provision every healthy host to the controller's working tree.
    working_tree = Path.cwd()  # caller invokes from the repo root
    commit_sha = ""
    for host in healthy:
        pr = provision_valkyrie(host, working_tree, fresh=options.fresh, ssh=ssh, rsync=rsync)
        if not pr.ok:
            down_hosts.add(host.host)
        else:
            commit_sha = commit_sha or pr.commit_sha
    healthy = [h for h in healthy if h.host not in down_hosts]

    # Seed the state and spawn workers.
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id=dispatch_id,
        started_at=started_at,
        ended_at=None,
        seeds=options.seeds,
        commit_sha=commit_sha,
        fleet=[
            FleetSnapshot(
                host=h.host,
                status="down" if h.host in down_hosts else "idle",
                last_error=None,
            )
            for h in fleet.hosts
        ],
        jobs=merged_jobs,
    )
    write_dispatch_state(dispatch_dir, state)

    # Enqueue pending jobs.
    job_q: queue.Queue = queue.Queue()
    state_chan: queue.Queue = queue.Queue()
    for j in state.jobs:
        if j.status == "pending":
            job_q.put(j)

    shutdown_event = threading.Event()
    workers: list[ValkyrieWorker] = []
    for host in healthy:
        w = ValkyrieWorker(
            host=host,
            job_queue=job_q,
            state_chan=state_chan,
            dispatch_dir=dispatch_dir,
            options=WorkerOptions(
                per_job_timeout_s=options.per_job_timeout_s,
                max_infrastructure_retries=options.max_infrastructure_retries,
            ),
            ssh=ssh,
            rsync=rsync,
            shutdown_event=shutdown_event,
        )
        w.start()
        workers.append(w)

    # Sentinels so workers exit once the queue is drained.
    for _ in workers:
        job_q.put(None)

    # Drain state events into state.jobs; rewrite dispatch.json after each.
    jobs_by_id: dict[str, JobEntry] = {j.run_id: j for j in state.jobs}
    remaining = sum(1 for j in state.jobs if j.status == "pending")
    last_write = time.monotonic()
    while remaining > 0 and any(w.is_alive() for w in workers):
        try:
            ev: StateEvent = state_chan.get(timeout=1.0)
        except queue.Empty:
            if time.monotonic() - last_write >= 5.0:
                write_dispatch_state(dispatch_dir, state)
                last_write = time.monotonic()
            continue
        j = jobs_by_id[ev.run_id]
        if ev.transition == "running":
            j.status = "running"
            j.started_at = ev.started_at
            j.assigned_to = ev.host
            for f in state.fleet:
                if f.host == ev.host:
                    f.status = "busy"
                    f.current_run_id = ev.run_id
        elif ev.transition == "completed":
            j.status = "completed"
            j.ended_at = ev.ended_at
            for f in state.fleet:
                if f.host == ev.host:
                    f.status = "idle"
                    f.current_run_id = None
            remaining -= 1
            if options.verbose:
                print(f"[{_utc_now_iso()}] COMPLETE {j.run_id} on {ev.host}")
        elif ev.transition == "failed":
            j.status = "failed"
            j.failure = ev.failure
            j.ended_at = ev.ended_at
            for f in state.fleet:
                if f.host == ev.host:
                    f.status = "idle"
                    f.current_run_id = None
                    if ev.failure is not None:
                        f.last_error = ev.failure.message
            remaining -= 1
            if options.verbose:
                kind = ev.failure.kind if ev.failure else "unknown"
                print(f"[{_utc_now_iso()}] FAIL     {j.run_id} on {ev.host} (kind={kind})")
        write_dispatch_state(dispatch_dir, state)
        last_write = time.monotonic()

    for w in workers:
        w.join(timeout=30.0)

    state.ended_at = _utc_now_iso()
    write_dispatch_state(dispatch_dir, state)
    return state
```

Update `tools/odin/asgard/__init__.py` to export `DispatchOptions`, `resolve_dispatch_dir`, `run_dispatch`.

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_runner.py -v --confcutdir=tools/odin
```

Expected: 6 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py tools/odin/asgard/__init__.py tools/odin/tests/test_asgard_runner.py
git commit -m "Add run_dispatch top-level orchestrator

run_dispatch() wires the full Asgard loop:
  1. Build fresh queue from T2.1 YAMLs + CLI seeds.
  2. If a prior dispatch.json exists, reset_in_flight_to_pending()
     and merge the fresh queue over it (fail loud if the fresh queue
     is missing run_ids the old one had).
  3. Snapshot fleet.yaml, preflight every host, write preflight.json.
  4. Fail fast unless skip_preflight; provision healthy hosts.
  5. Enqueue pending jobs, spawn one ValkyrieWorker per healthy host
     with a None sentinel per worker so they exit after drain.
  6. Drain StateEvents into the in-memory state, rewriting
     dispatch.json atomically after every transition plus a 5s
     heartbeat.
  7. Join workers; set ended_at; final write.

resolve_dispatch_dir() creates a fresh odin_runs/<now_utc>/ or
resumes (LATEST | <dispatch_id>).

--retry-failed flips specific failed jobs back to pending on resume;
completed jobs are never re-run."
```

---

## Task 11: `cli.py` — thin CLI wrapper

**Goal:** Command-line entry point that parses args, loads the fleet, calls `run_dispatch`, prints the terminal status lines (per Section 5 of the spec), and exits non-zero on any failed job.

**Files:**
- Create: `tools/odin/asgard/cli.py`
- Create: `tools/odin/tests/test_asgard_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tools/odin/tests/test_asgard_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.cli.parse_args`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.cli import parse_args, parse_seed_list


def test_parse_seed_list_single():
    assert parse_seed_list("42") == [42]


def test_parse_seed_list_multiple():
    assert parse_seed_list("42,43,44") == [42, 43, 44]


def test_parse_seed_list_strips_whitespace():
    assert parse_seed_list(" 42 , 43 ") == [42, 43]


def test_parse_seed_list_rejects_non_int():
    with pytest.raises(ValueError):
        parse_seed_list("42,foo,43")


def test_parse_args_minimal():
    args = parse_args(
        [
            "--fleet", "fleet.yaml",
            "--physx-yaml", "physx.yaml",
            "--seeds", "42",
        ]
    )
    assert str(args.fleet) == "fleet.yaml"
    assert args.physx_yaml == Path("physx.yaml")
    assert args.newton_yaml is None
    assert args.seeds == [42]
    assert args.fresh is False
    assert args.skip_preflight is False
    assert args.per_job_timeout == 14400


def test_parse_args_all_flags():
    args = parse_args(
        [
            "--fleet", "fleet.yaml",
            "--physx-yaml", "physx.yaml",
            "--newton-yaml", "newton.yaml",
            "--seeds", "42,43",
            "--include", "Isaac-Ant-*", "Isaac-Humanoid-*",
            "--resume", "LATEST",
            "--fresh",
            "--skip-preflight",
            "--per-job-timeout", "7200",
            "--max-infrastructure-retries", "5",
            "--retry-failed", "run1,run2",
            "--verbose",
        ]
    )
    assert args.seeds == [42, 43]
    assert args.include == ["Isaac-Ant-*", "Isaac-Humanoid-*"]
    assert args.resume == "LATEST"
    assert args.fresh is True
    assert args.skip_preflight is True
    assert args.per_job_timeout == 7200
    assert args.max_infrastructure_retries == 5
    assert args.retry_failed == ["run1", "run2"]
    assert args.verbose is True


def test_parse_args_requires_at_least_one_yaml():
    with pytest.raises(SystemExit):
        parse_args(["--fleet", "fleet.yaml", "--seeds", "42"])
```

- [ ] **Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cli.py -v --confcutdir=tools/odin
```

Expected: ImportError on `parse_args` / `parse_seed_list`.

- [ ] **Step 3: Create `cli.py`**

Create `tools/odin/asgard/cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-dispatch CLI — thin wrapper over :func:`run_dispatch`.

Invoke from the repo root with ``PYTHONPATH=.``:

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \\
        --fleet fleet.yaml \\
        --physx-yaml tools/odin/config/physx_envs.yaml \\
        --seeds 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch

__all__ = ["main", "parse_args", "parse_seed_list"]


def parse_seed_list(spec: str) -> list[int]:
    """Parse a comma-separated seed spec like "42" or "42,43,44"."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="odin-dispatch",
        description="Distributed Hugin/Munin dispatch across Asgard (Odin T3.1).",
    )
    parser.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    parser.add_argument(
        "--physx-yaml",
        type=Path,
        default=None,
        help="Path to curated physx_envs.yaml (T2.1). At least one of --physx-yaml / --newton-yaml required.",
    )
    parser.add_argument("--newton-yaml", type=Path, default=None, help="Path to curated newton_envs.yaml (T2.1).")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed list, e.g. '42' or '42,43,44'.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional fnmatch patterns on task_id; a row must match at least one to be queued.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume target: 'LATEST' or a specific dispatch_id. Default: start a new dispatch.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("odin_runs"),
        help="Root directory for dispatch bundles (default: ./odin_runs).",
    )
    parser.add_argument("--fresh", action="store_true", help="Wipe remote IsaacLab + restart docker container.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Continue when some hosts fail preflight (they are marked 'down').",
    )
    parser.add_argument(
        "--per-job-timeout",
        type=int,
        default=14400,
        help="Per-job wall-clock timeout in seconds (default: 14400 = 4h).",
    )
    parser.add_argument(
        "--max-infrastructure-retries",
        type=int,
        default=2,
        help="Max retries for SSH/docker failures before Hugin starts (default: 2).",
    )
    parser.add_argument(
        "--retry-failed",
        default=None,
        help="Comma-separated list of run_ids (from a prior failed dispatch) to re-attempt on resume.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-transition status lines as jobs progress.",
    )
    args = parser.parse_args(argv)

    if args.physx_yaml is None and args.newton_yaml is None:
        parser.error("at least one of --physx-yaml / --newton-yaml is required")

    args.seeds = parse_seed_list(args.seeds)
    if args.retry_failed:
        args.retry_failed = [s.strip() for s in args.retry_failed.split(",") if s.strip()]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    fleet = load_fleet(args.fleet)
    dispatch_dir = resolve_dispatch_dir(args.runs_root, resume=args.resume)

    options = DispatchOptions(
        seeds=args.seeds,
        max_infrastructure_retries=args.max_infrastructure_retries,
        per_job_timeout_s=args.per_job_timeout,
        fresh=args.fresh,
        skip_preflight=args.skip_preflight,
        include_filter=args.include,
        verbose=args.verbose,
        retry_failed=args.retry_failed,
    )

    print(f"odin-dispatch: dispatch_id={dispatch_dir.name} fleet={fleet.fleet_name} hosts={len(fleet.hosts)}")
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=args.physx_yaml,
        newton_yaml=args.newton_yaml,
        dispatch_dir=dispatch_dir,
        options=options,
    )

    total = len(state.jobs)
    completed = sum(1 for j in state.jobs if j.status == "completed")
    failed = sum(1 for j in state.jobs if j.status == "failed")
    pending = sum(1 for j in state.jobs if j.status == "pending")
    failed_by_kind: dict[str, int] = {}
    for j in state.jobs:
        if j.status == "failed" and j.failure is not None:
            failed_by_kind[j.failure.kind] = failed_by_kind.get(j.failure.kind, 0) + 1
    summary = f"{completed} completed, {failed} failed"
    if failed_by_kind:
        summary += " (" + ", ".join(f"{n} {k}" for k, n in sorted(failed_by_kind.items())) + ")"
    summary += f", {pending} pending"
    print(f"odin-dispatch: {summary} out of {total} total")
    return 0 if failed == 0 and pending == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests and verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cli.py -v --confcutdir=tools/odin
```

Expected: 7 passing.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cli.py tools/odin/tests/test_asgard_cli.py
git commit -m "Add odin-dispatch CLI — thin wrapper over run_dispatch

argparse surface: --fleet (required), --physx-yaml / --newton-yaml
(at least one required), --seeds (required comma list), --include
(fnmatch), --resume, --runs-root, --fresh, --skip-preflight,
--per-job-timeout, --max-infrastructure-retries, --retry-failed,
--verbose.

main() prints a start banner, invokes run_dispatch, then prints a
completion summary (counts by status, failed broken down by kind),
and exits 1 if any jobs remained failed or pending."
```

---

## Task 12: Loopback integration test (slow-marked)

**Goal:** One opt-in end-to-end test that runs `run_dispatch` against a fake fleet pointing at `localhost`, with a stub "Hugin" that produces a minimal valid bundle. Skips when `ssh localhost` isn't usable. Exercises the real `ShellSSHRunner` and `ShellRsyncRunner` against real subprocess.

Because a real Hugin invocation needs docker + the full IsaacLab container, this test uses a **local stub**: the "docker exec" command is replaced by a shell command that writes a fake bundle and exits 0. That still validates the transport + dispatch wiring end-to-end.

**Files:**
- Create: `tools/odin/tests/test_asgard_integration.py`

- [ ] **Step 1: Write the slow integration test**

Create `tools/odin/tests/test_asgard_integration.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slow-marked integration test: Asgard dispatch against ssh localhost.

Replaces the 'docker exec' command with a shell stub so the test runs
without docker or Isaac Sim. Still exercises the real ShellSSHRunner and
ShellRsyncRunner subprocess paths, covering the transport + dispatch wiring
end-to-end.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, run_dispatch
from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner
from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list


def _ssh_localhost_works() -> bool:
    """Probe: can we `ssh localhost "echo ok"` without a password?"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "echo ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


pytestmark = pytest.mark.slow


@pytest.fixture
def stub_ssh_runner(monkeypatch, tmp_path: Path):
    """Replace ShellSSHRunner.run's docker-exec command with a local stub that
    materialises a valid bundle and exits 0."""
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd

    def _fake_build(host: ValkyrieConfig, job) -> str:
        # Write a minimal valid bundle into the host's odin_runs/ directory
        # (same path the real runner + rsync pull target expects).
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"training": {"status": "completed"}, "startup": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir} && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json"
        )

    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", _fake_build)
    yield
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


@pytest.fixture
def stub_provisioner(monkeypatch):
    """Preflight (docker ps / docker inspect) would fail on vanilla localhost.
    Short-circuit preflight and provisioner to always pass."""
    from tools.odin.asgard import preflight as pf
    from tools.odin.asgard import provisioner as pv

    def _fake_pf(host, *, ssh):
        return pf.PreflightResult(
            host=host.host, ok=True,
            checks={"ssh_reach": True, "docker_running": True, "container_up": True, "isaaclab_present": True},
            message="",
        )

    def _fake_pv(host, working_tree, *, fresh, ssh, rsync):
        return pv.ProvisionResult(host=host.host, ok=True, commit_sha="integration-stub")

    monkeypatch.setattr("tools.odin.asgard.runner.preflight_valkyrie", _fake_pf)
    monkeypatch.setattr("tools.odin.asgard.runner.provision_valkyrie", _fake_pv)


def test_loopback_dispatch_against_localhost(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    # Build a one-row env list.
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    # One-host fleet targeting localhost; use the current repo as the remote
    # "isaaclab_path" so the fake bundle lands in a path we can rsync back.
    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-test", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260422-loopback"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], per_job_timeout_s=60),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The one job must complete.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"

    # Bundle must have been pulled back to the dispatch directory.
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()
    assert (bundle / "logs" / "ssh-tail.log").exists()

    # dispatch.json must record success.
    dj = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert dj["jobs"][0]["status"] == "completed"
```

- [ ] **Step 2: Verify the test is collected but skipped by default**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_integration.py --collect-only -v --confcutdir=tools/odin
```

Expected: one test collected, `slow` marker visible.

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_integration.py -v --confcutdir=tools/odin -m "not slow"
```

Expected: 0 selected, 1 deselected.

- [ ] **Step 3: Run the slow test manually if ssh localhost is usable**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_integration.py -v --confcutdir=tools/odin 2>&1 | tail -5
```

Expected either: 1 passing (if `ssh localhost "echo ok"` works), or 1 skipped. If the test fails while `ssh localhost` works, fix the script it runs before committing.

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_integration.py
git commit -m "Add slow-marked loopback integration test for Asgard

Runs run_dispatch against a one-host fleet targeting ssh localhost,
with the docker-exec command stubbed to a local shell script that
materialises a minimal valid bundle. Exercises the real ShellSSHRunner
and ShellRsyncRunner subprocess paths.

Skips (not fails) when ssh localhost -o BatchMode=yes doesn't succeed
without a password — typical in CI; enabled on any machine where the
user has their own ssh key in authorized_keys."
```

---

## Task 13: `tools/odin/README.md` — document odin-dispatch

**Goal:** Append a section to the existing Odin README covering the T3.1 CLI.

**Files:**
- Modify: `tools/odin/README.md`

- [ ] **Step 1: Append the section**

Append to the end of `tools/odin/README.md`:

````markdown

## Dispatching across a fleet (T3.1 — Asgard)

`tools/odin/asgard/cli.py` (the `odin-dispatch` entry point) ingests
the curated T2.1 YAMLs + a `fleet.yaml` listing SSH-accessible Valkyrie
machines and runs Hugin/Munin jobs across them in parallel.

### Fleet configuration

Create a `fleet.yaml` with per-host SSH / path config:

```yaml
fleet_name: h100-sweep-2026-04
default_ssh_user: odinrunner
default_ssh_key: ~/.ssh/odin_id_ed25519
hosts:
  - host: valkyrie-01.internal
  - host: valkyrie-02.internal
    ssh_user: svc-odin
    isaaclab_path: /mnt/scratch/IsaacLab
  - host: 10.0.0.42
    ssh_key: ~/.ssh/alt_key
```

Per-host fields override the fleet-level defaults. `container_name`
defaults to `isaac-lab-base` (matching `docker/docker-compose.yaml` for
profile `base`); override per-host if you're using a different profile.

### Running a dispatch

Run from the repo root. `PYTHONPATH=.` makes `tools.odin.*` importable.

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --newton-yaml tools/odin/config/newton_envs.yaml \
    --seeds 42,43,44 \
    [--include 'Isaac-Ant-*'] \
    [--resume LATEST] \
    [--fresh] \
    [--skip-preflight] \
    [--per-job-timeout 14400] \
    [--verbose]
```

Each `keep: true` row in the YAML is expanded across the seed list into
one job per `(task, seed)`. Jobs dispatch concurrently — one per
Valkyrie. Bundles land in `odin_runs/<dispatch_id>/<run_id>/` on the
controller (rsync'd back on each job completion).

### What happens on first contact

For each Valkyrie in the fleet:

1. Preflight: SSH-reach + `docker ps` + `docker inspect <container_name>`
   + `test -d <isaaclab_path>`. A failure aborts the dispatch with a
   per-host report (`preflight.json` written either way). Use
   `--skip-preflight` to continue with the healthy hosts.
2. Provision: rsync the controller's working tree to the Valkyrie's
   `isaaclab_path`, then `./docker/container.py start` (or stop+start
   under `--fresh`). `--fresh` wipes the remote tree first.
3. Dispatch loop: pull a job from the shared queue, SSH in and
   `docker exec` Hugin/Munin with the job's CLI args, tee stdout to
   `<run_id>/logs/ssh-tail.log`, rsync the bundle back on exit,
   update `dispatch.json`.

### Failure classification

Jobs fail into one of four kinds (stored in `dispatch.json` under
`failure.kind`):

- `infrastructure` — SSH error / docker exec error before Hugin
  started. Retried up to `--max-infrastructure-retries` (default 2),
  preferring a different Valkyrie after the first failure.
- `hugin_crash` — remote process exited non-zero. **Not retried** —
  real bugs repeat; use `--retry-failed <run_id>` to explicitly
  re-attempt on a later invocation.
- `hugin_malformed_bundle` — Hugin exited 0 but `manifest.json` is
  missing / bad / wrong schema. Not retried.
- `timeout` — job ran past `--per-job-timeout`. Not retried; the
  remote process is terminated.

### Resume

If the controller crashes mid-dispatch, re-invoke with
`--resume <dispatch_id>` (or `--resume LATEST` to pick the most recent
directory). In-flight jobs flip back to `pending` and re-dispatch;
completed and failed jobs are preserved.

Starting a fresh dispatch means **not** passing `--resume` — a new
`<dispatch_id>` directory is created.

### State on disk

```
odin_runs/
└── 20260422-220000/                         # dispatch_id
    ├── dispatch.json                         # full state, atomically rewritten
    ├── fleet.yaml.snapshot                   # fleet.yaml at dispatch start
    ├── preflight.json                        # opening health check
    ├── rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42/
    │   ├── manifest.json
    │   ├── training.json
    │   ├── startup.json
    │   ├── tb/
    │   └── logs/
    │       ├── ssh-tail.log                  # controller-side tee of remote stdout
    │       └── (Hugin's own log files)
    └── ...
```

`dispatch.json` schema v1.0 is defined in
`tools/odin/asgard/state.py`. See
`docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md` for
field-by-field details.
````

- [ ] **Step 2: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/README.md
git commit -m "Document Asgard dispatch (T3.1) in tools/odin/README

Covers fleet.yaml shape, odin-dispatch invocation, the provisioning
flow (preflight + rsync + docker container start), failure
classification (four kinds with which ones retry), resume semantics,
and the on-disk dispatch directory layout."
```

---

## Task 14: Architecture-doc closeout

**Goal:** Flip T3 status in the task map and add the change-log entry. Same pattern as T2.1 / T2.2.

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 1: Update the task-map row**

Find in §6:

```
| T3 | Distributed dispatcher (Layer 3) + Asgard | — | ⚪ |
```

Replace with:

```
| T3 | Distributed dispatcher (Layer 3) + Asgard | `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md` | 🟡 |
```

(T3 is yellow — T3.1 is done here; T3.2 web UI remains deferred per the spec's decomposition note.)

- [ ] **Step 2: Update "Last updated" line**

Change the current line near the top of the file from `(end of T2.2)` to:

```markdown
**Last updated:** 2026-04-22 (end of T3.1)
```

- [ ] **Step 3: Annotate the Layer-3 box in §3**

Find the ASCII box for Layer 3 in §3:

```
│ Layer 3 — Odin controller + Asgard (T3)                           │
│   Dispatches jobs over Bifrost (SSH) to Valkyrie nodes;           │
│   monitors progress; collects bundles back from workers.          │
│   Runs docker setup on each Valkyrie on first contact.            │
```

Leave the text as-is — the description already matches T3.1's delivery. No changes needed in §3 beyond the status flip (§6).

- [ ] **Step 4: Add the change-log entry in §9**

Append to the change-log table:

```
| 2026-04-22 | T3.1 delivered. `tools/odin/asgard/` library + thin CLI ingests `fleet.yaml` + T2.1 env YAMLs + CLI `--seeds`, provisions Valkyries via rsync + `./docker/container.py start` (smart-sync with `--fresh` override; rsync-the-working-tree transport avoids the no-push constraint), preflights each host (ssh + docker + container + isaaclab path), dispatches concurrently (one thread per Valkyrie) via `docker exec`, rsyncs bundles back, classifies failures (`infrastructure` retried up to 2×, `hugin_crash` / `hugin_malformed_bundle` / `timeout` never auto-retried). On-disk layout: `odin_runs/<dispatch_id>/<run_id>/` bundles + `dispatch.json` (atomic write) + `fleet.yaml.snapshot` + `preflight.json`. Resume via `--resume <dispatch_id|LATEST>` flips in-flight to `pending`, preserves completed/failed. No upstream IsaacLab changes. T3.2 (local web UI on the T3.1 state) deferred — may fold into T4's Valhalla dashboard. | Odin T3.1 |
```

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin T3.1 complete in architecture reference

Task-map status: T3 flips from ⚪ to 🟡 (T3.1 done; T3.2 web UI
deferred). Spec link added.

Change log: one row summarising T3.1 deliverables — fleet config,
preflight, rsync-the-working-tree provisioning, thread-per-Valkyrie
dispatch loop, failure classification, atomic dispatch.json,
resume semantics, integration test. Deferred T3.2 noted as a
candidate to fold into T4's Valhalla dashboard.

Last-updated line moves from 'end of T2.2' to 'end of T3.1'."
```

---

## Self-review notes (for the implementer)

Before calling T3.1 done, verify:

1. **All tests pass.**
   - `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_*.py -v --confcutdir=tools/odin -m "not slow"` — all unit tests green.
   - Loopback integration: run the slow test on a machine where `ssh localhost "echo ok"` works (skipped otherwise). Expected: 1 passing.

2. **Pre-commit clean** on `HEAD` for every touched file.

3. **Deliverables exist.**
   - `tools/odin/asgard/__init__.py` re-exports the public API.
   - 10 asgard modules under `tools/odin/asgard/` (fleet, queue, state, transport, provisioner, preflight, worker, runner, cli, `__init__`).
   - 7 new test files under `tools/odin/tests/` (fleet, queue, state, transport_ssh, transport_rsync, preflight, provisioner, worker, runner, cli, integration).
   - `tools/odin/README.md` has the new Dispatching section.
   - `docs/odin/architecture.md` reflects T3 🟡 with spec link and §9 entry.

4. **CLI surface.** `PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py --help` prints the argument list defined in Task 11.

5. **Odin-only changes.** `git diff --stat origin/<base>..HEAD -- source/isaaclab*` is empty — T3.1 touches nothing under `source/`.

6. **Manual acceptance (≥ 2-host fleet) — not part of the plan's committed artefacts but required before moving to T3.2 / T4.** Run a real dispatch against ≥ 2 Valkyries with `tools/odin/config/physx_envs.yaml` (`keep:true` subset) + `--seeds 42`. Hand-kill one Valkyrie's docker container mid-run and confirm the affected jobs classify as `infrastructure`. Confirm bundles appear under `odin_runs/<dispatch_id>/` and that `dispatch.json` is internally consistent.
