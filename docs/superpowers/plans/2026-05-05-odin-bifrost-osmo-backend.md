# Odin Bifrost (OSMO backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tools/odin/bifrost/`, a peer dispatch path that submits Odin eval jobs as a single OSMO workflow with N parallel tasks, sharing run-id/manifest/bundle layout with the existing `tools/odin/asgard/` path while leaving asgard untouched.

**Architecture:** One OSMO workflow per dispatch; one OSMO task per `(task, seed)` row. Bundles come back via `osmo dataset download` into `odin_runs/<dispatch_id>/<run_id>/`, where Valhalla picks them up unchanged. No host-lifecycle plumbing — OSMO subsumes preflight, image pull, scheduling, retries (`exitActions`), output upload, and log streaming. Failure mapping table reduces OSMO's terminal-state taxonomy to Odin's four kinds (`infrastructure` / `hugin_crash` / `hugin_malformed_bundle` / `timeout`).

**Tech Stack:** Python 3, Jinja2 (already a transitive dep), PyYAML, subprocess. No new required dependencies. The `osmo` CLI is invoked via subprocess — no HTTP client. Tests use stdlib `unittest.mock` to patch `subprocess.run`.

**Reference spec:** [`docs/superpowers/specs/2026-05-05-odin-bifrost-osmo-backend-design.md`](../specs/2026-05-05-odin-bifrost-osmo-backend-design.md). Read it before starting; this plan assumes you have. Section numbers below (e.g. "spec §5.1") refer to that file.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `tools/odin/bifrost/__init__.py` | Package marker. Empty besides SPDX header. |
| `tools/odin/bifrost/config.py` | Load + validate `bifrost-osmo.yaml`. Typed errors. |
| `tools/odin/bifrost/workflow.py` | `osmo_safe_task_name`, row builder, Jinja render, tarball staging. |
| `tools/odin/bifrost/templates/dispatch.yaml.j2` | Jinja template — one workflow, N parallel tasks. |
| `tools/odin/bifrost/client.py` | `osmo` CLI subprocess wrappers + typed errors. |
| `tools/odin/bifrost/poller.py` | OSMO_STATE_TO_FAILURE_KIND table + poll loop + dispatch.json writer. |
| `tools/odin/bifrost/bundle.py` | `osmo dataset download` + manifest validation + idempotent re-download. |
| `tools/odin/bifrost/cli.py` | `odin-bifrost-dispatch` entry point. |
| `tools/odin/tests/test_bifrost_config.py` | Config validation tests. |
| `tools/odin/tests/test_bifrost_workflow.py` | DNS-safe-name + template-render tests. |
| `tools/odin/tests/test_bifrost_client.py` | Subprocess wrapper tests. |
| `tools/odin/tests/test_bifrost_poller.py` | Failure-kind mapping + state machine tests. |
| `tools/odin/tests/test_bifrost_bundle.py` | Download + idempotency tests. |
| `tools/odin/tests/test_bifrost_cli.py` | Argparse + dry-run + resume + retry-failed tests. |
| `tools/odin/tests/test_bifrost_integration.py` | Slow-marked, env-flag-gated end-to-end test. |
| `tools/odin/config/bifrost-osmo.yaml.example` | Reference config, committed to repo. |

**Modified files:**

| Path | Change |
|---|---|
| `tools/odin/asgard/state.py` | Bump `SCHEMA_VERSION` to `"1.5"`; add optional fields (`dispatcher`, `osmo_workflow_id`, `parent_dispatch_id` at top level; `osmo_task_name` per job). |
| `tools/odin/asgard/jobs.py` | Add `osmo_task_name: str | None = None` to `JobEntry`. |
| `tools/odin/README.md` | New section: "Dispatching to OSMO (Bifrost)". |

**Out of scope for this plan** (per spec §2):

- No mixed-backend dispatches.
- No shared retry SQLite (`tools/odin/valhalla/dashboard/retry_cli.py` left alone for now).
- No multi-node OSMO `groups`.
- No custom Odin docker image build.
- No dashboard OSMO link rendering.
- No `asgard → horde` rename.

---

## Conventions

- **TDD** — every task writes the failing test first, runs it to confirm failure, implements, runs again, commits.
- **Run tests with:**
  - Fast unit tests: `PYTHONPATH=. python3 -m pytest tools/odin/tests/<file> -v --confcutdir=tools/odin`
  - Slow integration: same pattern but `-m slow` and with `ODIN_OSMO_INTEGRATION=1`.
- **Pre-commit before each commit**: `./isaaclab.sh -f`. If hooks modify files, stage and re-run before committing.
- **Commit messages**: imperative subject ≤50 chars, no trailing period, no AI co-author lines (per `AGENTS.md`).
- **Branch**: keep working on `antoiner/feat/odin`. No PR yet.

---

## Phase 0 — Schema bump

The schema bump is the foundation: every later task either writes to the bumped schema or reads it. Doing it first means we never have to re-stamp anything.

### Task 1: Bump dispatch.json schema to v1.5 (additive)

**Files:**
- Modify: `tools/odin/asgard/jobs.py` (add `osmo_task_name` field to `JobEntry`)
- Modify: `tools/odin/asgard/state.py` (bump `SCHEMA_VERSION`, extend `DispatchState`, plumb new fields through serializers)
- Test: `tools/odin/tests/test_asgard_state.py` (existing file if it exists, else create)

- [ ] **Step 1: Find existing state tests**

```bash
ls tools/odin/tests/test_asgard_state*.py 2>/dev/null
grep -rln "SCHEMA_VERSION\|read_dispatch_state\|write_dispatch_state" tools/odin/tests/
```

If no `test_asgard_state.py` exists, create one. Otherwise extend it.

- [ ] **Step 2: Write failing test for schema v1.5 round-trip**

In `tools/odin/tests/test_asgard_state.py`, add:

```python
import json
from pathlib import Path

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
    DispatchState,
    read_dispatch_state,
    write_dispatch_state,
)


def test_schema_version_is_1_5():
    assert SCHEMA_VERSION == "1.5"


def test_dispatch_state_round_trip_with_osmo_fields(tmp_path: Path):
    job = JobEntry(
        run_id="rsl-rl_physx_X_seed42",
        task_id="X",
        framework="rsl-rl",
        backend="physx",
        num_envs=4096,
        max_iterations=500,
        seed=42,
        bundle_dir_name="rsl-rl_physx_X_seed42",
        status="completed",
        osmo_task_name="rsl-rl-physx-x-seed42",
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260505-150000",
        started_at="2026-05-05T15:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="deadbeef",
        fleet=[],
        jobs=[job],
        dispatcher="osmo",
        osmo_workflow_id="odin-disp-20260505-150000-1",
        parent_dispatch_id=None,
    )
    write_dispatch_state(tmp_path, state)
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.dispatcher == "osmo"
    assert loaded.osmo_workflow_id == "odin-disp-20260505-150000-1"
    assert loaded.parent_dispatch_id is None
    assert loaded.jobs[0].osmo_task_name == "rsl-rl-physx-x-seed42"


def test_dispatch_state_back_compat_loads_v1_4_without_dispatcher(tmp_path: Path):
    """An old dispatch.json with no `dispatcher` field loads with dispatcher='asgard'."""
    payload = {
        "schema_version": "1.4",
        "dispatch_id": "20260101-000000",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "",
        "fleet": [],
        "jobs": [],
        "skipped": [],
        "quarantined_hosts": [],
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload))
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.dispatcher == "asgard"
    assert loaded.osmo_workflow_id is None
    assert loaded.parent_dispatch_id is None
```

- [ ] **Step 3: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin
```

Expected: FAIL on `assert SCHEMA_VERSION == "1.5"` and on missing `dispatcher` / `osmo_task_name` fields.

- [ ] **Step 4: Update `JobEntry` to add `osmo_task_name`**

In `tools/odin/asgard/jobs.py`, find the `JobEntry` dataclass and add the field. Example diff:

```python
@dataclass
class JobEntry:
    run_id: str
    task_id: str
    framework: str
    backend: str  # physx | newton
    num_envs: int
    max_iterations: int
    seed: int
    bundle_dir_name: str
    status: str = "pending"
    assigned_to: str | None = None
    attempts: int = 0
    failure: FailureInfo | None = None
    preferred_not: set[str] = field(default_factory=set)
    started_at: str | None = None
    ended_at: str | None = None
    per_job_timeout_s: int | None = None
    osmo_task_name: str | None = None  # NEW
```

- [ ] **Step 5: Bump SCHEMA_VERSION and extend DispatchState**

In `tools/odin/asgard/state.py`:

```python
SCHEMA_VERSION = "1.5"
```

Extend `DispatchState`:

```python
@dataclass
class DispatchState:
    schema_version: str
    dispatch_id: str
    started_at: str
    ended_at: str | None
    seeds: list[int]
    commit_sha: str
    fleet: list[FleetSnapshot]
    jobs: list[JobEntry]
    skipped: list[SkippedEntry] = field(default_factory=list)
    quarantined_hosts: list[QuarantinedHost] = field(default_factory=list)
    dispatcher: str = "asgard"  # NEW; "asgard" | "osmo"
    osmo_workflow_id: str | None = None  # NEW
    parent_dispatch_id: str | None = None  # NEW
```

- [ ] **Step 6: Plumb new fields through `_state_to_dict` / `_state_from_dict` / `_job_to_dict` / `_job_from_dict`**

```python
def _job_to_dict(j: JobEntry) -> dict[str, Any]:
    d: dict[str, Any] = {
        "run_id": j.run_id,
        # ... existing fields ...
        "per_job_timeout_s": j.per_job_timeout_s,
        "osmo_task_name": j.osmo_task_name,  # NEW
    }
    # ... failure handling unchanged ...
    return d


def _job_from_dict(d: dict[str, Any]) -> JobEntry:
    # ... existing parsing ...
    return JobEntry(
        # ... existing kwargs ...
        per_job_timeout_s=d.get("per_job_timeout_s"),
        osmo_task_name=d.get("osmo_task_name"),  # NEW; None for old v1.4
    )


def _state_to_dict(s: DispatchState) -> dict[str, Any]:
    return {
        "schema_version": s.schema_version,
        "dispatch_id": s.dispatch_id,
        # ... existing fields ...
        "quarantined_hosts": [...],
        "dispatcher": s.dispatcher,                    # NEW
        "osmo_workflow_id": s.osmo_workflow_id,        # NEW
        "parent_dispatch_id": s.parent_dispatch_id,    # NEW
    }


def _state_from_dict(d: dict[str, Any]) -> DispatchState:
    # ... existing schema check + parsing ...
    return DispatchState(
        # ... existing kwargs ...
        quarantined_hosts=[...],
        dispatcher=str(d.get("dispatcher") or "asgard"),     # NEW; default "asgard"
        osmo_workflow_id=d.get("osmo_workflow_id"),          # NEW
        parent_dispatch_id=d.get("parent_dispatch_id"),      # NEW
    )
```

- [ ] **Step 7: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_asgard_state.py -v --confcutdir=tools/odin
```

Expected: PASS, all three tests green.

- [ ] **Step 8: Run the full asgard test suite to confirm no regressions**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/ -v --confcutdir=tools/odin -k "not slow"
```

Expected: PASS. If any asgard test fails because it asserts the old schema version, update its expectation to `"1.5"` (these are factual updates, not behavior changes).

- [ ] **Step 9: Update CHANGELOG**

Per `AGENTS.md`, the schema lives in `tools/odin/asgard/`. There is no `source/<package>/docs/CHANGELOG.rst` entry needed for `tools/odin/` (it's not an extension). Skip the changelog update; this is a tools-tree change.

- [ ] **Step 10: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/state.py tools/odin/asgard/jobs.py tools/odin/tests/test_asgard_state.py
git commit -m "asgard: bump dispatch.json schema to v1.5 with optional osmo fields"
```

---

## Phase 1 — Bifrost scaffolding

### Task 2: Create empty bifrost package

**Files:**
- Create: `tools/odin/bifrost/__init__.py`
- Create: `tools/odin/bifrost/templates/__init__.py` (so the directory is a package — Jinja loader path-based, but tests still need to import the package)

- [ ] **Step 1: Create the package files**

`tools/odin/bifrost/__init__.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin Bifrost — OSMO-backed dispatch path.

Peer to :mod:`tools.odin.asgard`. Bifrost submits eval jobs as a
single OSMO workflow with N parallel tasks; bundles return via
``osmo dataset download`` into the canonical
``odin_runs/<dispatch_id>/<run_id>/`` layout.

See ``docs/superpowers/specs/2026-05-05-odin-bifrost-osmo-backend-design.md``.
"""
```

`tools/odin/bifrost/templates/__init__.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

- [ ] **Step 2: Verify the package imports cleanly**

```bash
PYTHONPATH=. python3 -c "import tools.odin.bifrost; print(tools.odin.bifrost.__doc__[:60])"
```

Expected: prints the first 60 chars of the docstring.

- [ ] **Step 3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/__init__.py tools/odin/bifrost/templates/__init__.py
git commit -m "bifrost: add empty package skeleton"
```

---

### Task 3: Implement `bifrost/config.py` — load + validate

**Files:**
- Create: `tools/odin/bifrost/config.py`
- Create: `tools/odin/tests/test_bifrost_config.py`

- [ ] **Step 1: Write failing test**

`tools/odin/tests/test_bifrost_config.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

from tools.odin.bifrost.config import BifrostConfig, BifrostConfigError, load_bifrost_config


VALID_YAML = """\
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL
image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
  pull_credential: ngc-readonly
defaults:
  resources:
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000
  exec_timeout: 14400
  queue_timeout: 7200
retry:
  reschedule_codes: "3001-3006"
  restart_codes: ""
bundle_dataset_prefix: odin
code_delivery:
  mode: files_upload
  source_root: tools/odin
"""


def test_load_valid_config(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML)
    cfg = load_bifrost_config(p)
    assert isinstance(cfg, BifrostConfig)
    assert cfg.pool == "rtx-pro-6000-eval"
    assert cfg.image.reference == "nvcr.io/nvidia/isaac-lab:2.2.0"
    assert cfg.image.pull_credential == "ngc-readonly"
    assert cfg.defaults.resources.gpu == 1
    assert cfg.code_delivery.mode == "files_upload"
    assert cfg.priority == "NORMAL"


def test_priority_must_be_in_enum(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("priority: NORMAL", "priority: URGENT"))
    with pytest.raises(BifrostConfigError, match="priority"):
        load_bifrost_config(p)


def test_code_delivery_mode_must_be_in_enum(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("mode: files_upload", "mode: ftp_upload"))
    with pytest.raises(BifrostConfigError, match="code_delivery.mode"):
        load_bifrost_config(p)


def test_missing_required_field_raises(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("pool: rtx-pro-6000-eval\n", ""))
    with pytest.raises(BifrostConfigError, match="pool"):
        load_bifrost_config(p)


def test_pull_credential_optional(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("  pull_credential: ngc-readonly\n", ""))
    cfg = load_bifrost_config(p)
    assert cfg.image.pull_credential is None


def test_resources_must_have_all_keys(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("    platform: rtx-pro-6000\n", ""))
    with pytest.raises(BifrostConfigError, match="resources.platform"):
        load_bifrost_config(p)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_config.py -v --confcutdir=tools/odin
```

Expected: FAIL with `ImportError: cannot import name 'BifrostConfig'`.

- [ ] **Step 3: Implement `bifrost/config.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and validate ``bifrost-osmo.yaml``.

See spec §5.1 for the schema. All errors raise :class:`BifrostConfigError`
with a key path so the operator knows which field to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BifrostConfig",
    "BifrostConfigError",
    "ImageSpec",
    "ResourcesSpec",
    "DefaultsSpec",
    "RetrySpec",
    "CodeDeliverySpec",
    "load_bifrost_config",
]


_PRIORITIES = {"HIGH", "NORMAL", "LOW"}
_CODE_DELIVERY_MODES = {"files_upload", "rsync", "image_baked"}


class BifrostConfigError(ValueError):
    """Raised when ``bifrost-osmo.yaml`` fails validation."""


@dataclass(frozen=True)
class ImageSpec:
    reference: str
    pull_credential: str | None


@dataclass(frozen=True)
class ResourcesSpec:
    cpu: int
    gpu: int
    memory: str
    storage: str
    platform: str


@dataclass(frozen=True)
class DefaultsSpec:
    resources: ResourcesSpec
    exec_timeout: int
    queue_timeout: int


@dataclass(frozen=True)
class RetrySpec:
    reschedule_codes: str
    restart_codes: str


@dataclass(frozen=True)
class CodeDeliverySpec:
    mode: str  # files_upload | rsync | image_baked
    source_root: str


@dataclass(frozen=True)
class BifrostConfig:
    osmo_profile: str
    pool: str
    priority: str  # HIGH | NORMAL | LOW
    image: ImageSpec
    defaults: DefaultsSpec
    retry: RetrySpec
    bundle_dataset_prefix: str
    code_delivery: CodeDeliverySpec


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise BifrostConfigError(f"missing required field: {ctx}.{key}" if ctx else f"missing required field: {key}")
    return d[key]


def _require_str(d: dict[str, Any], key: str, ctx: str) -> str:
    v = _require(d, key, ctx)
    if not isinstance(v, str) or not v:
        raise BifrostConfigError(f"{ctx}.{key} must be a non-empty string")
    return v


def _require_int(d: dict[str, Any], key: str, ctx: str) -> int:
    v = _require(d, key, ctx)
    if not isinstance(v, int) or isinstance(v, bool):
        raise BifrostConfigError(f"{ctx}.{key} must be an integer")
    return v


def load_bifrost_config(path: Path) -> BifrostConfig:
    """Load and validate ``bifrost-osmo.yaml``.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        A fully populated :class:`BifrostConfig`.

    Raises:
        BifrostConfigError: If any required field is missing or invalid.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise BifrostConfigError("top-level YAML must be a mapping")

    osmo_profile = _require_str(raw, "osmo_profile", "")
    pool = _require_str(raw, "pool", "")
    priority = _require_str(raw, "priority", "")
    if priority not in _PRIORITIES:
        raise BifrostConfigError(f"priority must be one of {sorted(_PRIORITIES)}; got {priority!r}")

    image_d = _require(raw, "image", "")
    if not isinstance(image_d, dict):
        raise BifrostConfigError("image must be a mapping")
    image = ImageSpec(
        reference=_require_str(image_d, "reference", "image"),
        pull_credential=image_d.get("pull_credential"),
    )

    defaults_d = _require(raw, "defaults", "")
    if not isinstance(defaults_d, dict):
        raise BifrostConfigError("defaults must be a mapping")
    res_d = _require(defaults_d, "resources", "defaults")
    if not isinstance(res_d, dict):
        raise BifrostConfigError("defaults.resources must be a mapping")
    resources = ResourcesSpec(
        cpu=_require_int(res_d, "cpu", "defaults.resources"),
        gpu=_require_int(res_d, "gpu", "defaults.resources"),
        memory=_require_str(res_d, "memory", "defaults.resources"),
        storage=_require_str(res_d, "storage", "defaults.resources"),
        platform=_require_str(res_d, "platform", "defaults.resources"),
    )
    defaults = DefaultsSpec(
        resources=resources,
        exec_timeout=_require_int(defaults_d, "exec_timeout", "defaults"),
        queue_timeout=_require_int(defaults_d, "queue_timeout", "defaults"),
    )

    retry_d = raw.get("retry") or {}
    retry = RetrySpec(
        reschedule_codes=str(retry_d.get("reschedule_codes") or ""),
        restart_codes=str(retry_d.get("restart_codes") or ""),
    )

    bundle_dataset_prefix = _require_str(raw, "bundle_dataset_prefix", "")

    cd_d = _require(raw, "code_delivery", "")
    if not isinstance(cd_d, dict):
        raise BifrostConfigError("code_delivery must be a mapping")
    cd_mode = _require_str(cd_d, "mode", "code_delivery")
    if cd_mode not in _CODE_DELIVERY_MODES:
        raise BifrostConfigError(
            f"code_delivery.mode must be one of {sorted(_CODE_DELIVERY_MODES)}; got {cd_mode!r}"
        )
    code_delivery = CodeDeliverySpec(
        mode=cd_mode,
        source_root=_require_str(cd_d, "source_root", "code_delivery"),
    )

    return BifrostConfig(
        osmo_profile=osmo_profile,
        pool=pool,
        priority=priority,
        image=image,
        defaults=defaults,
        retry=retry,
        bundle_dataset_prefix=bundle_dataset_prefix,
        code_delivery=code_delivery,
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_config.py -v --confcutdir=tools/odin
```

Expected: PASS, all six tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/config.py tools/odin/tests/test_bifrost_config.py
git commit -m "bifrost: add bifrost-osmo.yaml loader + validator"
```

---

### Task 4: Implement `osmo_safe_task_name` helper

**Files:**
- Create: `tools/odin/bifrost/workflow.py` (start with just this helper; we'll grow it)
- Create: `tools/odin/tests/test_bifrost_workflow.py`

- [ ] **Step 1: Write failing test**

`tools/odin/tests/test_bifrost_workflow.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

import pytest

from tools.odin.bifrost.workflow import osmo_safe_task_name

DNS_1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def test_simple_run_id_lowercased_and_dashed():
    out = osmo_safe_task_name("rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42")
    assert DNS_1123_LABEL.match(out), f"not DNS-1123-safe: {out!r}"
    assert out.startswith("rsl-rl-physx-isaac-ant-direct-v0")


def test_no_underscores_in_output():
    out = osmo_safe_task_name("a_b_c")
    assert "_" not in out


def test_no_dots_in_output():
    out = osmo_safe_task_name("a.b.c")
    assert "." not in out


def test_truncation_appends_hash():
    long = "x" * 80
    out = osmo_safe_task_name(long)
    assert len(out) <= 63
    assert DNS_1123_LABEL.match(out)
    # Two different long inputs should produce different outputs
    other = osmo_safe_task_name("y" * 80)
    assert out != other


def test_no_leading_or_trailing_dash():
    out = osmo_safe_task_name("_foo_")
    assert not out.startswith("-")
    assert not out.endswith("-")


def test_idempotent_on_safe_name():
    safe = "rsl-rl-physx-x-seed42"
    assert osmo_safe_task_name(safe) == safe
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_workflow.py -v --confcutdir=tools/odin
```

Expected: FAIL with `ImportError: cannot import name 'osmo_safe_task_name'`.

- [ ] **Step 3: Implement `bifrost/workflow.py` — initial stub**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workflow rendering for Bifrost.

This module:
- builds DNS-1123-safe OSMO task names from Odin run_ids
  (:func:`osmo_safe_task_name`),
- (later) renders the Jinja workflow template from a row list.
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["osmo_safe_task_name"]


_DNS_1123_LABEL_MAX = 63
_HASH_SUFFIX_LEN = 7  # "-" + 6 hex chars
_NON_ALNUM_DASH = re.compile(r"[^a-z0-9-]")
_RUN_OF_DASHES = re.compile(r"-+")


def osmo_safe_task_name(run_id: str) -> str:
    """Convert an Odin ``run_id`` into a DNS-1123-compliant OSMO task name.

    Constraints (per Kubernetes' DNS-1123 label rules):

    - At most 63 characters.
    - Only lowercase alphanumerics and ``-``.
    - Must not start or end with ``-``.

    On truncation, a 6-hex-char hash of the full ``run_id`` is appended so
    distinct long inputs produce distinct outputs.

    Args:
        run_id: Odin run_id (e.g. ``rsl-rl_physx_Isaac-Ant_seed42``).

    Returns:
        A DNS-1123-safe label.
    """
    lowered = run_id.lower()
    dashed = re.sub(r"[_.\s]+", "-", lowered)
    only_safe = _NON_ALNUM_DASH.sub("-", dashed)
    collapsed = _RUN_OF_DASHES.sub("-", only_safe).strip("-")
    if not collapsed:
        # Degenerate input: emit a stable hash-only label.
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
        return f"odin-{digest}"
    if len(collapsed) <= _DNS_1123_LABEL_MAX:
        return collapsed
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
    keep = _DNS_1123_LABEL_MAX - _HASH_SUFFIX_LEN
    return f"{collapsed[:keep].rstrip('-')}-{digest}"
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_workflow.py -v --confcutdir=tools/odin
```

Expected: PASS, all six tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/workflow.py tools/odin/tests/test_bifrost_workflow.py
git commit -m "bifrost: add osmo_safe_task_name DNS-1123 helper"
```

---

### Task 5: Add the Jinja workflow template

**Files:**
- Create: `tools/odin/bifrost/templates/dispatch.yaml.j2`
- Modify: `tools/odin/bifrost/workflow.py` (add `render_workflow_yaml`)
- Modify: `tools/odin/tests/test_bifrost_workflow.py` (add render tests)

- [ ] **Step 1: Write failing render tests**

Append to `tools/odin/tests/test_bifrost_workflow.py`:

```python
import yaml

from tools.odin.bifrost.config import (
    BifrostConfig,
    CodeDeliverySpec,
    DefaultsSpec,
    ImageSpec,
    ResourcesSpec,
    RetrySpec,
)
from tools.odin.bifrost.workflow import RenderRow, render_workflow_yaml


def _cfg(mode: str = "files_upload") -> BifrostConfig:
    return BifrostConfig(
        osmo_profile="prod",
        pool="rtx-pro-6000-eval",
        priority="NORMAL",
        image=ImageSpec(reference="nvcr.io/nvidia/isaac-lab:2.2.0", pull_credential="ngc-readonly"),
        defaults=DefaultsSpec(
            resources=ResourcesSpec(
                cpu=16, gpu=1, memory="64Gi", storage="64Gi", platform="rtx-pro-6000"
            ),
            exec_timeout=14400,
            queue_timeout=7200,
        ),
        retry=RetrySpec(reschedule_codes="3001-3006", restart_codes=""),
        bundle_dataset_prefix="odin",
        code_delivery=CodeDeliverySpec(mode=mode, source_root="tools/odin"),
    )


def _row(seed: int = 42, framework: str = "rsl-rl") -> RenderRow:
    return RenderRow(
        run_id=f"{framework}_physx_Isaac-Ant-Direct-v0_20260505-150000_seed{seed}",
        osmo_task_name=f"{framework}-physx-isaac-ant-seed{seed}".replace("_", "-"),
        framework=framework,
        framework_runner="hugin" if framework == "rsl-rl" else "munin",
        task_id="Isaac-Ant-Direct-v0",
        backend="physx",
        seed=seed,
        num_envs=4096,
        max_iterations=500,
    )


def test_render_workflow_yaml_is_valid_yaml_with_one_task():
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000",
        rows=[_row()],
        cfg=cfg,
        tarball_path="/tmp/odin-source.tar.gz",
    )
    parsed = yaml.safe_load(out)
    wf = parsed["workflow"]
    assert wf["name"] == "odin-disp-20260505-150000"
    assert wf["pool"] == "rtx-pro-6000-eval"
    assert len(wf["tasks"]) == 1
    task = wf["tasks"][0]
    assert task["image"] == "nvcr.io/nvidia/isaac-lab:2.2.0"
    assert task["credentials"]["registry"] == "ngc-readonly"
    assert task["outputs"][0]["dataset"]["name"] == "odin-20260505-150000-rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42"
    assert task["exitActions"]["RESCHEDULE"] == "3001-3006"


def test_render_workflow_yaml_n_parallel_tasks():
    cfg = _cfg()
    rows = [_row(seed=42), _row(seed=43, framework="skrl")]
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=rows, cfg=cfg, tarball_path="/tmp/odin-source.tar.gz"
    )
    parsed = yaml.safe_load(out)
    assert len(parsed["workflow"]["tasks"]) == 2


def test_render_workflow_special_token_output_survives_render():
    """OSMO's `{{output}}` must appear literally in the rendered YAML."""
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/tmp/odin-source.tar.gz"
    )
    assert "{{output}}" in out


def test_render_workflow_files_upload_mode_includes_tarball():
    cfg = _cfg(mode="files_upload")
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/abs/odin-source.tar.gz"
    )
    parsed = yaml.safe_load(out)
    files = parsed["workflow"]["tasks"][0]["files"]
    paths = [f.get("path") for f in files]
    assert "/workspace/odin-source.tar.gz" in paths
    tarball_entry = [f for f in files if f.get("path") == "/workspace/odin-source.tar.gz"][0]
    assert tarball_entry["localpath"] == "/abs/odin-source.tar.gz"


def test_render_workflow_rsync_mode_omits_tarball():
    cfg = _cfg(mode="rsync")
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path=None
    )
    parsed = yaml.safe_load(out)
    paths = [f.get("path") for f in parsed["workflow"]["tasks"][0]["files"]]
    assert "/workspace/odin-source.tar.gz" not in paths
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_workflow.py -v --confcutdir=tools/odin
```

Expected: FAIL on missing `RenderRow` and `render_workflow_yaml` imports.

- [ ] **Step 3: Create the Jinja template**

`tools/odin/bifrost/templates/dispatch.yaml.j2`:

```jinja
workflow:
  name: odin-disp-{{ dispatch_id }}
  pool: {{ cfg.pool }}
  resources:
    default:
      cpu: {{ cfg.defaults.resources.cpu }}
      gpu: {{ cfg.defaults.resources.gpu }}
      memory: {{ cfg.defaults.resources.memory }}
      storage: {{ cfg.defaults.resources.storage }}
      platform: {{ cfg.defaults.resources.platform }}
  timeouts:
    exec: {{ cfg.defaults.exec_timeout }}
    queue: {{ cfg.defaults.queue_timeout }}
  tasks:
{% for row in rows %}
  - name: {{ row.osmo_task_name }}
    image: {{ cfg.image.reference }}
{% if cfg.image.pull_credential %}
    credentials:
      registry: {{ cfg.image.pull_credential }}
{% endif %}
    environment:
      ACCEPT_EULA: "Y"
      NO_NUCLEUS: "Y"
      OMNI_KIT_ALLOW_ROOT: "1"
      ODIN_DISPATCH_ID: "{{ dispatch_id }}"
      ODIN_RUN_ID: "{{ row.run_id }}"
    command: ["bash"]
    args: ["/tmp/entry.sh"]
    files:
    - path: /tmp/entry.sh
      contents: |
        set -euxo pipefail
{% if cfg.code_delivery.mode == "files_upload" %}
        tar -xzf /workspace/odin-source.tar.gz -C /workspace/IsaacLab
{% endif %}
        cd /workspace/IsaacLab
        PYTHONPATH=. ./isaaclab.sh -p tools/odin/{{ row.framework_runner }}/run.py \
          --task {{ row.task_id }} \
          --backend {{ row.backend }} \
          --seed {{ row.seed }} \
          --num_envs {{ row.num_envs }} \
          --max_iterations {{ row.max_iterations }} \
          --runs_root '{{ '{{output}}' }}'
{% if cfg.code_delivery.mode == "files_upload" %}
    - localpath: {{ tarball_path }}
      path: /workspace/odin-source.tar.gz
{% endif %}
    outputs:
    - dataset:
        name: {{ cfg.bundle_dataset_prefix }}-{{ dispatch_id }}-{{ row.run_id }}
{% if cfg.retry.reschedule_codes or cfg.retry.restart_codes %}
    exitActions:
{% if cfg.retry.reschedule_codes %}
      RESCHEDULE: {{ cfg.retry.reschedule_codes }}
{% endif %}
{% if cfg.retry.restart_codes %}
      RESTART: {{ cfg.retry.restart_codes }}
{% endif %}
{% endif %}
{% endfor %}
```

- [ ] **Step 4: Add `RenderRow` + `render_workflow_yaml` to `bifrost/workflow.py`**

Append to `tools/odin/bifrost/workflow.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.odin.bifrost.config import BifrostConfig

__all__ = [
    "osmo_safe_task_name",
    "RenderRow",
    "render_workflow_yaml",
]


_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class RenderRow:
    """One row of the workflow render context.

    Maps 1:1 to a single task in the OSMO workflow YAML.
    """

    run_id: str
    osmo_task_name: str
    framework: str  # rsl-rl | skrl
    framework_runner: str  # hugin | munin
    task_id: str  # gym task id, e.g. Isaac-Ant-Direct-v0
    backend: str  # physx | newton
    seed: int
    num_envs: int
    max_iterations: int


def render_workflow_yaml(
    *,
    dispatch_id: str,
    rows: list[RenderRow],
    cfg: BifrostConfig,
    tarball_path: str | None,
) -> str:
    """Render the OSMO workflow YAML for one dispatch.

    Args:
        dispatch_id: Odin dispatch id (``YYYYMMDD-HHMMSS``).
        rows: One per ``(task, seed)`` to dispatch.
        cfg: Validated config from :func:`load_bifrost_config`.
        tarball_path: Controller-local path to the source tarball; required
            when ``cfg.code_delivery.mode == "files_upload"``, ignored
            otherwise.

    Returns:
        The rendered workflow YAML as a string. Caller writes it to disk
        and passes the path to ``osmo workflow submit``.
    """
    if cfg.code_delivery.mode == "files_upload" and not tarball_path:
        raise ValueError("tarball_path is required when code_delivery.mode == files_upload")
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("dispatch.yaml.j2")
    return template.render(
        dispatch_id=dispatch_id,
        rows=rows,
        cfg=cfg,
        tarball_path=tarball_path,
    )
```

- [ ] **Step 5: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_workflow.py -v --confcutdir=tools/odin
```

Expected: PASS, all eleven tests green (six DNS-name + five render).

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/workflow.py tools/odin/bifrost/templates/dispatch.yaml.j2 tools/odin/tests/test_bifrost_workflow.py
git commit -m "bifrost: render OSMO workflow YAML from row list + config"
```

---

### Task 6: Add tarball staging helper

**Files:**
- Modify: `tools/odin/bifrost/workflow.py` (add `stage_source_tarball`)
- Modify: `tools/odin/tests/test_bifrost_workflow.py` (add tarball test)

- [ ] **Step 1: Write failing test**

Append to `tools/odin/tests/test_bifrost_workflow.py`:

```python
import tarfile
from pathlib import Path

from tools.odin.bifrost.workflow import stage_source_tarball


def test_stage_source_tarball_produces_readable_archive(tmp_path: Path):
    src = tmp_path / "src"
    (src / "tools" / "odin").mkdir(parents=True)
    (src / "tools" / "odin" / "hello.py").write_text("print('hi')\n")
    out = tmp_path / "src.tar.gz"
    stage_source_tarball(src / "tools" / "odin", out, repo_root=src)
    assert out.exists()
    with tarfile.open(out, "r:gz") as t:
        names = t.getnames()
    assert "tools/odin/hello.py" in names
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_workflow.py::test_stage_source_tarball_produces_readable_archive -v --confcutdir=tools/odin
```

Expected: FAIL on missing import.

- [ ] **Step 3: Implement `stage_source_tarball`**

Append to `tools/odin/bifrost/workflow.py`:

```python
import tarfile

__all__ = [
    "osmo_safe_task_name",
    "RenderRow",
    "render_workflow_yaml",
    "stage_source_tarball",
]


def stage_source_tarball(source_dir: Path, dest_tarball: Path, *, repo_root: Path) -> None:
    """Tar ``source_dir`` into ``dest_tarball`` with paths relative to ``repo_root``.

    The OSMO entry script extracts with ``tar -xzf ... -C /workspace/IsaacLab``,
    so the tarball must contain paths like ``tools/odin/...`` (i.e. relative
    to the IsaacLab repo root, not absolute and not relative to ``source_dir``).

    Args:
        source_dir: Directory to tar (typically ``<repo_root>/tools/odin``).
        dest_tarball: Output path; parent dir created if missing.
        repo_root: The path under which archive entries should be relative.

    Raises:
        ValueError: If ``source_dir`` is not under ``repo_root``.
    """
    source_dir = source_dir.resolve()
    repo_root = repo_root.resolve()
    try:
        rel = source_dir.relative_to(repo_root)
    except ValueError as e:
        raise ValueError(f"source_dir {source_dir} not under repo_root {repo_root}") from e
    dest_tarball.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest_tarball, "w:gz") as tar:
        tar.add(source_dir, arcname=str(rel))
```

(Don't forget to update the existing `__all__` — replace the earlier list with the new four-entry one.)

- [ ] **Step 4: Run test — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_workflow.py -v --confcutdir=tools/odin
```

Expected: PASS, all twelve tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/workflow.py tools/odin/tests/test_bifrost_workflow.py
git commit -m "bifrost: add stage_source_tarball helper"
```

---

## Phase 2 — OSMO client + poller

### Task 7: Implement `bifrost/client.py` — submit + typed errors

**Files:**
- Create: `tools/odin/bifrost/client.py`
- Create: `tools/odin/tests/test_bifrost_client.py`

- [ ] **Step 1: Write failing tests for `submit`**

`tools/odin/tests/test_bifrost_client.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.odin.bifrost.client import (
    OsmoAuthError,
    OsmoCliError,
    OsmoClient,
    OsmoTransientError,
)


SUBMIT_STDOUT_OK = """\
Workflow submit successful.
Workflow ID        - my-wf-1
Workflow Overview  - https://osmo.example.com/workflows/my-wf-1
"""


def _completed(returncode=0, stdout="", stderr=""):
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def test_submit_parses_workflow_id_from_stdout(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout=SUBMIT_STDOUT_OK)) as run:
        wf_id = client.submit(yaml)
    assert wf_id == "my-wf-1"
    args, kwargs = run.call_args
    assert args[0][:3] == ["osmo", "workflow", "submit"]
    assert str(yaml) in args[0]
    assert kwargs.get("env", {}).get("OSMO_PROFILE") == "prod"


def test_submit_with_rsync_appends_flags(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout=SUBMIT_STDOUT_OK)) as run:
        client.submit(yaml, rsync_pairs=[("./tools/odin", "/workspace/odin-source")])
    cmd = run.call_args[0][0]
    assert "--rsync" in cmd
    rsync_idx = cmd.index("--rsync")
    assert cmd[rsync_idx + 1] == "./tools/odin:/workspace/odin-source"


def test_submit_raises_auth_error_on_401(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 401 Unauthorized")):
        with pytest.raises(OsmoAuthError):
            client.submit(yaml)


def test_submit_raises_transient_on_503(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 503 Service Unavailable")):
        with pytest.raises(OsmoTransientError):
            client.submit(yaml)


def test_submit_raises_generic_cli_error_on_other_failure(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="bad spec")):
        with pytest.raises(OsmoCliError, match="bad spec"):
            client.submit(yaml)


def test_submit_raises_when_id_unparseable(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout="weird output")):
        with pytest.raises(OsmoCliError, match="Workflow ID"):
            client.submit(yaml)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_client.py -v --confcutdir=tools/odin
```

Expected: FAIL on missing imports.

- [ ] **Step 3: Implement `bifrost/client.py` (submit only — other methods follow)**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin subprocess wrappers around the ``osmo`` CLI.

One module, one public class :class:`OsmoClient`. Each method shells out
to a single ``osmo`` invocation and parses the output. Errors are typed:

- :class:`OsmoAuthError` — auth/credential failure; caller surfaces.
- :class:`OsmoTransientError` — retryable (HTTP 5xx, connection reset).
- :class:`OsmoCliError` — anything else (bad spec, parse failure).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

__all__ = [
    "OsmoAuthError",
    "OsmoCliError",
    "OsmoClient",
    "OsmoTransientError",
]


class OsmoCliError(RuntimeError):
    """Generic ``osmo`` CLI failure (non-zero exit, parse failure, etc.)."""


class OsmoAuthError(OsmoCliError):
    """Auth failure (HTTP 401/403). Not retried."""


class OsmoTransientError(OsmoCliError):
    """Retryable failure (HTTP 5xx, connection issues)."""


_AUTH_PATTERN = re.compile(r"HTTP 40[13]|unauthori[sz]ed", re.IGNORECASE)
_TRANSIENT_PATTERN = re.compile(r"HTTP 5\d\d|connection (reset|refused|timed? out)", re.IGNORECASE)
_WORKFLOW_ID_PATTERN = re.compile(r"^Workflow ID\s+-\s+(\S+)", re.MULTILINE)


def _classify(stderr: str) -> type[OsmoCliError]:
    if _AUTH_PATTERN.search(stderr):
        return OsmoAuthError
    if _TRANSIENT_PATTERN.search(stderr):
        return OsmoTransientError
    return OsmoCliError


class OsmoClient:
    """Subprocess-based wrapper around the ``osmo`` CLI.

    Args:
        profile: OSMO profile name. Passed via ``OSMO_PROFILE`` env var on
            every invocation.
        executable: ``osmo`` binary path. Defaults to ``"osmo"`` (relies
            on ``$PATH``).
    """

    def __init__(self, *, profile: str, executable: str = "osmo") -> None:
        self._profile = profile
        self._exe = executable

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OSMO_PROFILE"] = self._profile
        return env

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def submit(self, yaml_path: Path, *, rsync_pairs: Iterable[tuple[str, str]] = ()) -> str:
        """Submit a workflow YAML and return the workflow_id.

        Args:
            yaml_path: Path to the rendered workflow YAML.
            rsync_pairs: Pairs of ``(local_path, container_path)`` for OSMO's
                ``--rsync`` continuous-sync feature.

        Returns:
            The OSMO workflow ID parsed from stdout.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cmd: list[str] = [self._exe, "workflow", "submit", str(yaml_path)]
        for local, remote in rsync_pairs:
            cmd.extend(["--rsync", f"{local}:{remote}"])
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow submit` failed: {cp.stderr.strip()}")
        m = _WORKFLOW_ID_PATTERN.search(cp.stdout)
        if not m:
            raise OsmoCliError(f"could not parse Workflow ID from osmo stdout: {cp.stdout!r}")
        return m.group(1)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_client.py -v --confcutdir=tools/odin
```

Expected: PASS, all six submit tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/client.py tools/odin/tests/test_bifrost_client.py
git commit -m "bifrost: add OsmoClient.submit with typed error classification"
```

---

### Task 8: Implement `OsmoClient.status` (JSON + table fallback)

**Files:**
- Modify: `tools/odin/bifrost/client.py`
- Modify: `tools/odin/tests/test_bifrost_client.py`

OSMO 6.2 may or may not support `osmo workflow status --output json`. Implement both paths: try JSON first, fall back to a simple table parser.

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_bifrost_client.py`:

```python
import json

from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot


STATUS_JSON_OK = json.dumps(
    {
        "id": "my-wf-1",
        "status": "RUNNING",
        "tasks": [
            {"name": "rsl-rl-physx-x-seed42", "status": "COMPLETED", "exit_code": 0},
            {"name": "rsl-rl-physx-x-seed43", "status": "FAILED", "exit_code": 137},
            {"name": "rsl-rl-physx-x-seed44", "status": "RUNNING", "exit_code": None},
        ],
    }
)

STATUS_TABLE_OK = """\
Workflow ID: my-wf-1
Status: RUNNING

Tasks:
NAME                       STATUS      EXIT
rsl-rl-physx-x-seed42      COMPLETED   0
rsl-rl-physx-x-seed43      FAILED      137
rsl-rl-physx-x-seed44      RUNNING     -
"""


def test_status_parses_json_output_when_available():
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout=STATUS_JSON_OK)) as run:
        snap = client.status("my-wf-1")
    cmd = run.call_args[0][0]
    assert "--output" in cmd and "json" in cmd
    assert isinstance(snap, WorkflowSnapshot)
    assert snap.workflow_id == "my-wf-1"
    assert snap.status == "RUNNING"
    assert len(snap.tasks) == 3
    completed = [t for t in snap.tasks if t.name.endswith("seed42")][0]
    assert completed.status == "COMPLETED"
    assert completed.exit_code == 0


def test_status_falls_back_to_table_parser_when_json_unsupported():
    """When `--output json` is unrecognized, retry without it and parse the table."""
    client = OsmoClient(profile="prod")
    json_attempt = _completed(returncode=2, stderr="unknown flag --output")
    table_attempt = _completed(stdout=STATUS_TABLE_OK)
    with patch("subprocess.run", side_effect=[json_attempt, table_attempt]) as run:
        snap = client.status("my-wf-1")
    assert run.call_count == 2
    assert snap.workflow_id == "my-wf-1"
    assert len(snap.tasks) == 3
    seed44 = [t for t in snap.tasks if t.name.endswith("seed44")][0]
    assert seed44.status == "RUNNING"
    assert seed44.exit_code is None


def test_status_raises_on_real_failure():
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 401 Unauthorized")):
        with pytest.raises(OsmoAuthError):
            client.status("my-wf-1")
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_client.py -v --confcutdir=tools/odin
```

Expected: FAIL on missing `TaskSnapshot` / `WorkflowSnapshot` / `client.status`.

- [ ] **Step 3: Add status types and method**

Append to `tools/odin/bifrost/client.py`:

```python
import json
from dataclasses import dataclass

__all__ = [
    "OsmoAuthError",
    "OsmoCliError",
    "OsmoClient",
    "OsmoTransientError",
    "TaskSnapshot",
    "WorkflowSnapshot",
]


@dataclass(frozen=True)
class TaskSnapshot:
    name: str
    status: str  # OSMO task status (COMPLETED, FAILED, RUNNING, ...)
    exit_code: int | None


@dataclass(frozen=True)
class WorkflowSnapshot:
    workflow_id: str
    status: str  # OSMO workflow status
    tasks: list[TaskSnapshot]


_UNKNOWN_FLAG_PATTERNS = (
    re.compile(r"unknown flag", re.IGNORECASE),
    re.compile(r"unrecognized argument", re.IGNORECASE),
    re.compile(r"--output.*not recognized", re.IGNORECASE),
)


def _looks_like_unknown_flag(stderr: str) -> bool:
    return any(p.search(stderr) for p in _UNKNOWN_FLAG_PATTERNS)
```

Then add the method on `OsmoClient`:

```python
    def status(self, workflow_id: str) -> WorkflowSnapshot:
        """Fetch the workflow snapshot.

        Tries ``--output json`` first; if the flag is unrecognized, retries
        with the default table output and parses that.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cmd_json = [self._exe, "workflow", "status", workflow_id, "--output", "json"]
        cp = self._run(cmd_json)
        if cp.returncode == 0:
            return self._parse_status_json(cp.stdout, workflow_id)
        if _looks_like_unknown_flag(cp.stderr):
            cp2 = self._run([self._exe, "workflow", "status", workflow_id])
            if cp2.returncode != 0:
                raise _classify(cp2.stderr)(f"`osmo workflow status` failed: {cp2.stderr.strip()}")
            return self._parse_status_table(cp2.stdout, workflow_id)
        raise _classify(cp.stderr)(f"`osmo workflow status` failed: {cp.stderr.strip()}")

    @staticmethod
    def _parse_status_json(stdout: str, workflow_id: str) -> WorkflowSnapshot:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise OsmoCliError(f"could not parse JSON status: {e}") from e
        tasks = [
            TaskSnapshot(
                name=str(t["name"]),
                status=str(t["status"]),
                exit_code=(None if t.get("exit_code") in (None, "-") else int(t["exit_code"])),
            )
            for t in data.get("tasks") or []
        ]
        return WorkflowSnapshot(
            workflow_id=str(data.get("id", workflow_id)),
            status=str(data["status"]),
            tasks=tasks,
        )

    @staticmethod
    def _parse_status_table(stdout: str, workflow_id: str) -> WorkflowSnapshot:
        wf_status = "UNKNOWN"
        tasks: list[TaskSnapshot] = []
        in_tasks = False
        for raw in stdout.splitlines():
            line = raw.strip()
            if line.startswith("Status:"):
                wf_status = line.split(":", 1)[1].strip()
            elif line.startswith("NAME"):
                in_tasks = True
                continue
            elif in_tasks and line:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name, status = parts[0], parts[1]
                exit_str = parts[2] if len(parts) >= 3 else "-"
                exit_code = None if exit_str in ("-", "") else int(exit_str)
                tasks.append(TaskSnapshot(name=name, status=status, exit_code=exit_code))
        return WorkflowSnapshot(workflow_id=workflow_id, status=wf_status, tasks=tasks)
```

(Update `__all__` once at the top so it includes the new `TaskSnapshot` and `WorkflowSnapshot`.)

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_client.py -v --confcutdir=tools/odin
```

Expected: PASS, all nine tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/client.py tools/odin/tests/test_bifrost_client.py
git commit -m "bifrost: add OsmoClient.status with JSON + table fallback"
```

---

### Task 9: Implement `OsmoClient.logs`, `dataset_download`, `cancel`

**Files:**
- Modify: `tools/odin/bifrost/client.py`
- Modify: `tools/odin/tests/test_bifrost_client.py`

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_bifrost_client.py`:

```python
def test_dataset_download_invokes_correct_command(tmp_path: Path):
    client = OsmoClient(profile="prod")
    dest = tmp_path / "bundle"
    with patch("subprocess.run", return_value=_completed()) as run:
        client.dataset_download("odin-disp1-run42", dest)
    cmd = run.call_args[0][0]
    assert cmd[:3] == ["osmo", "dataset", "download"]
    assert "odin-disp1-run42" in cmd
    assert str(dest) in cmd


def test_dataset_download_creates_dest_dir(tmp_path: Path):
    client = OsmoClient(profile="prod")
    dest = tmp_path / "nested" / "bundle"
    with patch("subprocess.run", return_value=_completed()):
        client.dataset_download("odin-disp1-run42", dest)
    assert dest.exists()


def test_dataset_download_raises_on_failure(tmp_path: Path):
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 503")):
        with pytest.raises(OsmoTransientError):
            client.dataset_download("odin-disp1-run42", tmp_path / "x")


def test_cancel_invokes_correct_command():
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed()) as run:
        client.cancel("my-wf-1")
    cmd = run.call_args[0][0]
    assert cmd == ["osmo", "workflow", "cancel", "my-wf-1"]


def test_logs_invokes_correct_command_no_follow():
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout="line1\nline2\n")) as run:
        out = list(client.logs("my-wf-1", "task-a", follow=False))
    cmd = run.call_args[0][0]
    assert cmd[:3] == ["osmo", "workflow", "logs"]
    assert "--follow" not in cmd
    assert b"line1" in out[0]
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_client.py -v --confcutdir=tools/odin
```

Expected: FAIL on missing methods.

- [ ] **Step 3: Add the three methods to `OsmoClient`**

```python
    def dataset_download(self, name: str, dest_dir: Path) -> None:
        """Download an OSMO dataset to a local directory.

        Creates ``dest_dir`` (and parents) if missing.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [self._exe, "dataset", "download", name, str(dest_dir)]
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo dataset download` failed: {cp.stderr.strip()}")

    def cancel(self, workflow_id: str) -> None:
        """Cancel an in-flight workflow."""
        cp = self._run([self._exe, "workflow", "cancel", workflow_id])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow cancel` failed: {cp.stderr.strip()}")

    def logs(self, workflow_id: str, task_name: str, *, follow: bool = False) -> "Iterator[bytes]":
        """Yield log bytes from the named task.

        With ``follow=False``, runs the command to completion and yields
        the captured stdout as a single bytes chunk. With ``follow=True``,
        streams stdout line-by-line until the subprocess exits.
        """
        cmd = [self._exe, "workflow", "logs", workflow_id, task_name]
        if follow:
            cmd.append("--follow")
            return self._stream_logs(cmd)
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow logs` failed: {cp.stderr.strip()}")
        return iter([cp.stdout.encode()])

    def _stream_logs(self, cmd: list[str]) -> "Iterator[bytes]":
        proc = subprocess.Popen(
            cmd, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, b""):
                yield line
        finally:
            proc.stdout.close()
            proc.wait()
```

(Add `from typing import Iterator` at the top if not already imported.)

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_client.py -v --confcutdir=tools/odin
```

Expected: PASS, all fourteen tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/client.py tools/odin/tests/test_bifrost_client.py
git commit -m "bifrost: add logs, dataset_download, cancel client methods"
```

---

### Task 10: Implement OSMO_STATE_TO_FAILURE_KIND mapping table

**Files:**
- Create: `tools/odin/bifrost/poller.py` (start with just the mapping)
- Create: `tools/odin/tests/test_bifrost_poller.py`

- [ ] **Step 1: Write failing test**

`tools/odin/tests/test_bifrost_poller.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from tools.odin.bifrost.poller import (
    TERMINAL_OSMO_STATES,
    classify_terminal_state,
    is_terminal,
)


@pytest.mark.parametrize(
    "osmo_state, expected_kind",
    [
        ("FAILED", "hugin_crash"),
        ("FAILED_EXEC_TIMEOUT", "timeout"),
        ("FAILED_BACKEND_ERROR", "infrastructure"),
        ("FAILED_PREEMPTED", "infrastructure"),
        ("FAILED_EVICTED", "infrastructure"),
        ("FAILED_IMAGE_PULL", "infrastructure"),
        ("FAILED_START_ERROR", "infrastructure"),
        ("FAILED_START_TIMEOUT", "infrastructure"),
        ("FAILED_QUEUE_TIMEOUT", "infrastructure"),
        ("FAILED_SERVER_ERROR", "infrastructure"),
        ("FAILED_CANCELED", "infrastructure"),
    ],
)
def test_failure_kind_for_each_failed_state(osmo_state: str, expected_kind: str):
    assert classify_terminal_state(osmo_state) == expected_kind


def test_completed_returns_none():
    """COMPLETED isn't a failure kind — caller decides hugin_malformed_bundle separately."""
    assert classify_terminal_state("COMPLETED") is None


def test_classify_unknown_state_defaults_to_infrastructure():
    assert classify_terminal_state("FAILED_NOVEL_THING") == "infrastructure"


def test_is_terminal_recognizes_completed_and_failed_family():
    assert is_terminal("COMPLETED")
    assert is_terminal("FAILED")
    assert is_terminal("FAILED_BACKEND_ERROR")


def test_is_terminal_excludes_in_flight_states():
    for s in ("PENDING", "WAITING", "PROCESSING", "SCHEDULING", "INITIALIZING", "RUNNING", "RESCHEDULED"):
        assert not is_terminal(s), s


def test_terminal_set_complete():
    """Sanity-check the terminal set contains all FAILED_* and COMPLETED."""
    expected_subset = {
        "COMPLETED",
        "FAILED",
        "FAILED_EXEC_TIMEOUT",
        "FAILED_BACKEND_ERROR",
        "FAILED_PREEMPTED",
        "FAILED_EVICTED",
        "FAILED_IMAGE_PULL",
        "FAILED_START_ERROR",
        "FAILED_START_TIMEOUT",
        "FAILED_QUEUE_TIMEOUT",
        "FAILED_SERVER_ERROR",
        "FAILED_CANCELED",
    }
    assert expected_subset <= TERMINAL_OSMO_STATES
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_poller.py -v --confcutdir=tools/odin
```

Expected: FAIL on missing imports.

- [ ] **Step 3: Implement `bifrost/poller.py` (mapping only)**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Poll an OSMO workflow and classify per-task terminal states.

Single source of truth for the OSMO-state → Odin-failure-kind mapping
lives in :data:`OSMO_STATE_TO_FAILURE_KIND` (spec §7).
"""

from __future__ import annotations

__all__ = [
    "OSMO_STATE_TO_FAILURE_KIND",
    "TERMINAL_OSMO_STATES",
    "classify_terminal_state",
    "is_terminal",
]


# Maps OSMO terminal task states to Odin failure kinds. Per spec §7.
# COMPLETED is intentionally absent: callers decide
# hugin_malformed_bundle vs success after manifest validation.
OSMO_STATE_TO_FAILURE_KIND: dict[str, str] = {
    "FAILED": "hugin_crash",
    "FAILED_EXEC_TIMEOUT": "timeout",
    "FAILED_BACKEND_ERROR": "infrastructure",
    "FAILED_PREEMPTED": "infrastructure",
    "FAILED_EVICTED": "infrastructure",
    "FAILED_IMAGE_PULL": "infrastructure",
    "FAILED_START_ERROR": "infrastructure",
    "FAILED_START_TIMEOUT": "infrastructure",
    "FAILED_QUEUE_TIMEOUT": "infrastructure",
    "FAILED_SERVER_ERROR": "infrastructure",
    "FAILED_CANCELED": "infrastructure",
}

TERMINAL_OSMO_STATES: frozenset[str] = frozenset({"COMPLETED", *OSMO_STATE_TO_FAILURE_KIND.keys()})


def is_terminal(osmo_state: str) -> bool:
    """Return True iff ``osmo_state`` is one of the known terminal task states."""
    return osmo_state in TERMINAL_OSMO_STATES


def classify_terminal_state(osmo_state: str) -> str | None:
    """Return the Odin failure kind for an OSMO terminal state, or ``None`` for COMPLETED.

    Unknown ``FAILED_*`` states default to ``"infrastructure"`` to keep behavior
    safe under OSMO version drift.
    """
    if osmo_state == "COMPLETED":
        return None
    if osmo_state in OSMO_STATE_TO_FAILURE_KIND:
        return OSMO_STATE_TO_FAILURE_KIND[osmo_state]
    if osmo_state.startswith("FAILED"):
        return "infrastructure"
    return None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_poller.py -v --confcutdir=tools/odin
```

Expected: PASS, all six tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/poller.py tools/odin/tests/test_bifrost_poller.py
git commit -m "bifrost: add OSMO-state to failure-kind mapping"
```

---

### Task 11: Implement `bifrost/bundle.py` — download + manifest validation

**Files:**
- Create: `tools/odin/bifrost/bundle.py`
- Create: `tools/odin/tests/test_bifrost_bundle.py`

- [ ] **Step 1: Confirm the manifest validator location**

```bash
grep -rln "validate_manifest\|read_manifest" tools/odin/common/ tools/odin/asgard/ | head
```

If `tools/odin/common/manifest.py` exposes a manifest validator, use it. If not, the bundle module accepts validation as a callable parameter so we can swap implementations without rewriting the call site.

- [ ] **Step 2: Write failing tests**

`tools/odin/tests/test_bifrost_bundle.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.odin.bifrost.bundle import BundleResult, download_and_validate_bundle


def _client_writing_manifest(content: dict, run_subdir: str = "rsl-rl_physx_X_seed42"):
    """Make a MagicMock OsmoClient whose dataset_download writes a manifest.json."""
    client = MagicMock()

    def fake_download(name: str, dest_dir: Path) -> None:
        run_dir = Path(dest_dir) / run_subdir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(content))

    client.dataset_download.side_effect = fake_download
    return client


def test_download_writes_to_expected_path(tmp_path: Path):
    valid = lambda p: True  # noqa: E731
    client = _client_writing_manifest({"schema": "v1"})
    res = download_and_validate_bundle(
        client=client,
        dataset_name="odin-disp1-rsl-rl_physx_X_seed42",
        dispatch_dir=tmp_path,
        run_id="rsl-rl_physx_X_seed42",
        validator=valid,
    )
    expected_dir = tmp_path / "rsl-rl_physx_X_seed42"
    assert res.bundle_dir == expected_dir
    assert (expected_dir / "manifest.json").exists()
    assert res.is_valid


def test_invalid_manifest_marks_malformed(tmp_path: Path):
    client = _client_writing_manifest({"missing": "fields"})
    res = download_and_validate_bundle(
        client=client,
        dataset_name="odin-disp1-rsl-rl_physx_X_seed42",
        dispatch_dir=tmp_path,
        run_id="rsl-rl_physx_X_seed42",
        validator=lambda p: False,
    )
    assert not res.is_valid


def test_idempotent_skips_redownload_when_manifest_present(tmp_path: Path):
    client = _client_writing_manifest({"schema": "v1"})
    # Pre-populate the bundle dir with a valid manifest.
    bundle_dir = tmp_path / "rsl-rl_physx_X_seed42"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"schema": "v1"}))
    res = download_and_validate_bundle(
        client=client,
        dataset_name="odin-disp1-rsl-rl_physx_X_seed42",
        dispatch_dir=tmp_path,
        run_id="rsl-rl_physx_X_seed42",
        validator=lambda p: True,
    )
    client.dataset_download.assert_not_called()
    assert res.is_valid
```

- [ ] **Step 3: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_bundle.py -v --confcutdir=tools/odin
```

Expected: FAIL on missing imports.

- [ ] **Step 4: Implement `bifrost/bundle.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bundle pull-back for Bifrost: ``osmo dataset download`` + manifest validation.

The OSMO task writes ``{{output}}/<run_id>/manifest.json`` (and friends);
the dataset uploaded to OSMO therefore contains ``<run_id>/manifest.json``
at the top level. Downloading the dataset into ``<dispatch_dir>/`` lands
the bundle at ``<dispatch_dir>/<run_id>/manifest.json`` — exactly Odin's
canonical layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

__all__ = ["BundleResult", "download_and_validate_bundle"]


class _DownloaderProto(Protocol):
    def dataset_download(self, name: str, dest_dir: Path) -> None: ...


@dataclass(frozen=True)
class BundleResult:
    bundle_dir: Path
    is_valid: bool


def download_and_validate_bundle(
    *,
    client: _DownloaderProto,
    dataset_name: str,
    dispatch_dir: Path,
    run_id: str,
    validator: Callable[[Path], bool],
) -> BundleResult:
    """Download a dataset into ``<dispatch_dir>/<run_id>/`` and validate the manifest.

    Idempotent: if ``<dispatch_dir>/<run_id>/manifest.json`` already exists
    AND the validator accepts it, the download is skipped.

    Args:
        client: An object with a ``dataset_download(name, dest)`` method.
        dataset_name: OSMO dataset name to download.
        dispatch_dir: Local directory containing the dispatch (e.g.
            ``odin_runs/<dispatch_id>``).
        run_id: Odin run_id; the bundle will land at ``dispatch_dir / run_id``.
        validator: A callable taking the bundle directory and returning
            ``True`` iff the manifest passes validation.

    Returns:
        :class:`BundleResult` with the bundle directory and validation outcome.
    """
    bundle_dir = dispatch_dir / run_id
    manifest = bundle_dir / "manifest.json"
    if manifest.exists() and validator(bundle_dir):
        return BundleResult(bundle_dir=bundle_dir, is_valid=True)
    client.dataset_download(dataset_name, dispatch_dir)
    is_valid = manifest.exists() and validator(bundle_dir)
    return BundleResult(bundle_dir=bundle_dir, is_valid=is_valid)
```

- [ ] **Step 5: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_bundle.py -v --confcutdir=tools/odin
```

Expected: PASS, all three tests green.

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/bundle.py tools/odin/tests/test_bifrost_bundle.py
git commit -m "bifrost: add idempotent bundle download + manifest validation"
```

---

### Task 12: Implement the poll loop

**Files:**
- Modify: `tools/odin/bifrost/poller.py`
- Modify: `tools/odin/tests/test_bifrost_poller.py`

This is the biggest piece. The poll loop iterates until all task states are terminal, on each iteration calling `client.status`, classifying changes, triggering bundle downloads on COMPLETED, and writing `dispatch.json` atomically.

- [ ] **Step 1: Write failing tests**

Append to `tools/odin/tests/test_bifrost_poller.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, call

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import DispatchState, SCHEMA_VERSION, read_dispatch_state
from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot
from tools.odin.bifrost.poller import poll_until_terminal


def _job(run_id: str, osmo_task_name: str, status: str = "pending") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="X",
        framework="rsl-rl",
        backend="physx",
        num_envs=4096,
        max_iterations=500,
        seed=42,
        bundle_dir_name=run_id,
        status=status,
        osmo_task_name=osmo_task_name,
    )


def _state(tmp_path: Path, jobs: list[JobEntry]) -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260505-150000",
        started_at="2026-05-05T15:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="",
        fleet=[],
        jobs=jobs,
        dispatcher="osmo",
        osmo_workflow_id="my-wf-1",
    )


def test_poll_marks_completed_and_failed(tmp_path: Path):
    jobs = [_job("run1", "task-1"), _job("run2", "task-2")]
    state = _state(tmp_path, jobs)
    client = MagicMock()
    client.status.side_effect = [
        WorkflowSnapshot(
            "my-wf-1",
            "RUNNING",
            [
                TaskSnapshot("task-1", "RUNNING", None),
                TaskSnapshot("task-2", "RUNNING", None),
            ],
        ),
        WorkflowSnapshot(
            "my-wf-1",
            "COMPLETED",
            [
                TaskSnapshot("task-1", "COMPLETED", 0),
                TaskSnapshot("task-2", "FAILED", 137),
            ],
        ),
    ]
    bundle_calls: list[str] = []

    def on_completed(job: JobEntry) -> None:
        bundle_calls.append(job.run_id)

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=on_completed,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"
    assert state.jobs[1].status == "failed"
    assert state.jobs[1].failure is not None and state.jobs[1].failure.kind == "hugin_crash"
    assert bundle_calls == ["run1"]
    # dispatch.json was rewritten at least once
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.jobs[0].status == "completed"


def test_poll_handles_unknown_failed_state_as_infrastructure(tmp_path: Path):
    jobs = [_job("run1", "task-1")]
    state = _state(tmp_path, jobs)
    client = MagicMock()
    client.status.return_value = WorkflowSnapshot(
        "my-wf-1", "FAILED", [TaskSnapshot("task-1", "FAILED_NOVEL", 9999)]
    )
    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].failure.kind == "infrastructure"


def test_poll_skips_unknown_task_names(tmp_path: Path):
    """OSMO returning a task name that's not in our state must not crash."""
    jobs = [_job("run1", "task-1")]
    state = _state(tmp_path, jobs)
    client = MagicMock()
    client.status.return_value = WorkflowSnapshot(
        "my-wf-1",
        "COMPLETED",
        [
            TaskSnapshot("task-1", "COMPLETED", 0),
            TaskSnapshot("task-unknown", "COMPLETED", 0),
        ],
    )
    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_poller.py -v --confcutdir=tools/odin
```

Expected: FAIL on `poll_until_terminal` not implemented.

- [ ] **Step 3: Implement `poll_until_terminal`**

Append to `tools/odin/bifrost/poller.py`:

```python
import time
from pathlib import Path
from typing import Callable, Protocol

from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.state import DispatchState, write_dispatch_state
from tools.odin.bifrost.client import WorkflowSnapshot

__all__ = [
    "OSMO_STATE_TO_FAILURE_KIND",
    "TERMINAL_OSMO_STATES",
    "classify_terminal_state",
    "is_terminal",
    "poll_until_terminal",
]


class _StatusClient(Protocol):
    def status(self, workflow_id: str) -> WorkflowSnapshot: ...


def _osmo_status_to_job_status(osmo_state: str) -> str:
    """Map an OSMO task state to the Odin job status string used in dispatch.json."""
    if osmo_state == "COMPLETED":
        return "completed"
    if osmo_state == "RUNNING":
        return "running"
    if is_terminal(osmo_state):
        return "failed"
    # SUBMITTING, WAITING, PROCESSING, SCHEDULING, INITIALIZING, RESCHEDULED.
    return "pending"


def poll_until_terminal(
    *,
    client: _StatusClient,
    state: DispatchState,
    dispatch_dir: Path,
    on_task_completed: Callable[[JobEntry], None],
    poll_interval_s: float,
) -> None:
    """Drive an OSMO workflow to completion, writing dispatch.json atomically.

    Args:
        client: Has ``status(workflow_id) -> WorkflowSnapshot``.
        state: Mutated in place. Must have ``osmo_workflow_id`` set.
        dispatch_dir: Where dispatch.json lives.
        on_task_completed: Called once per task that transitions to COMPLETED.
            Implementations typically enqueue a bundle download.
        poll_interval_s: Seconds between status calls. Set to 0 in tests.

    Returns when every job is in a terminal state.
    """
    if state.osmo_workflow_id is None:
        raise ValueError("state.osmo_workflow_id is required for OSMO polling")
    by_osmo_name = {j.osmo_task_name: j for j in state.jobs if j.osmo_task_name}
    completed_seen: set[str] = set()
    while not _all_terminal(state):
        snap = client.status(state.osmo_workflow_id)
        for task in snap.tasks:
            job = by_osmo_name.get(task.name)
            if job is None:
                continue  # Unknown task — log via state's general mechanism in caller.
            new_status = _osmo_status_to_job_status(task.status)
            if new_status == job.status:
                continue
            job.status = new_status
            if is_terminal(task.status):
                if task.status == "COMPLETED":
                    if job.run_id not in completed_seen:
                        completed_seen.add(job.run_id)
                        on_task_completed(job)
                else:
                    kind = classify_terminal_state(task.status) or "infrastructure"
                    job.failure = FailureInfo(
                        kind=kind,
                        message=f"OSMO task {task.name} terminal state {task.status} (exit={task.exit_code})",
                        details={"osmo_state": task.status, "exit_code": task.exit_code},
                    )
        write_dispatch_state(dispatch_dir, state)
        if _all_terminal(state):
            break
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)


def _all_terminal(state: DispatchState) -> bool:
    return all(j.status in ("completed", "failed") for j in state.jobs)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_poller.py -v --confcutdir=tools/odin
```

Expected: PASS, all nine tests green (six mapping + three poll-loop).

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/poller.py tools/odin/tests/test_bifrost_poller.py
git commit -m "bifrost: add OSMO workflow polling loop with atomic dispatch.json"
```

---

## Phase 3 — CLI

### Task 13: Implement `odin-bifrost-dispatch` CLI — argparse + planner + dry-run

**Files:**
- Create: `tools/odin/bifrost/cli.py`
- Create: `tools/odin/tests/test_bifrost_cli.py`
- Create: `tools/odin/config/bifrost-osmo.yaml.example`

- [ ] **Step 1: Find existing planner code in asgard to lift**

```bash
grep -n "physx_envs\|build_jobs\|expand.*seeds\|JobEntry(" tools/odin/asgard/cli.py tools/odin/asgard/queue.py 2>/dev/null | head -30
```

If asgard has a row-from-yaml builder we can import, use it. Otherwise we replicate the planner inline (small) — bifrost CLI does not get its own copy of the env-yaml validator; that's strictly a config concern.

- [ ] **Step 2: Write failing CLI tests**

`tools/odin/tests/test_bifrost_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def example_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "bifrost-osmo.yaml"
    cfg.write_text(
        "osmo_profile: prod\n"
        "pool: rtx-pro-6000-eval\n"
        "priority: NORMAL\n"
        "image:\n"
        "  reference: nvcr.io/nvidia/isaac-lab:2.2.0\n"
        "defaults:\n"
        "  resources:\n"
        "    cpu: 16\n"
        "    gpu: 1\n"
        "    memory: 64Gi\n"
        "    storage: 64Gi\n"
        "    platform: rtx-pro-6000\n"
        "  exec_timeout: 14400\n"
        "  queue_timeout: 7200\n"
        "retry: {reschedule_codes: '3001-3006', restart_codes: ''}\n"
        "bundle_dataset_prefix: odin\n"
        "code_delivery: {mode: files_upload, source_root: tools/odin}\n"
    )
    return cfg


@pytest.fixture
def example_physx_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "physx.yaml"
    p.write_text(
        "envs:\n"
        "- task_id: Isaac-Ant-Direct-v0\n"
        "  framework: rsl-rl\n"
        "  num_envs: 4096\n"
        "  max_iterations: 500\n"
        "  keep: true\n"
    )
    return p


def test_dry_run_writes_workflow_yaml_and_exits_zero(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
    from tools.odin.bifrost import cli as bifrost_cli

    runs_root = tmp_path / "odin_runs"
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(example_config),
            "--physx-yaml",
            str(example_physx_yaml),
            "--seeds",
            "42",
            "--runs-root",
            str(runs_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    # Dry-run should have created a dispatch dir with a rendered workflow.yaml.
    dispatch_dirs = list(runs_root.iterdir())
    assert len(dispatch_dirs) == 1
    assert (dispatch_dirs[0] / "workflow.yaml").exists()


def test_seed_expansion_creates_one_task_per_seed(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
    """Two seeds × one keep:true env → 2 tasks in the rendered workflow."""
    import yaml as y

    from tools.odin.bifrost import cli as bifrost_cli

    runs_root = tmp_path / "odin_runs"
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(example_config),
            "--physx-yaml",
            str(example_physx_yaml),
            "--seeds",
            "42,43",
            "--runs-root",
            str(runs_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    workflow_yaml = list(runs_root.iterdir())[0] / "workflow.yaml"
    parsed = y.safe_load(workflow_yaml.read_text())
    assert len(parsed["workflow"]["tasks"]) == 2
```

- [ ] **Step 3: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py -v --confcutdir=tools/odin
```

Expected: FAIL on `tools.odin.bifrost.cli` not yet importable.

- [ ] **Step 4: Implement `bifrost/cli.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``odin-bifrost-dispatch`` CLI entry point.

Submits a single OSMO workflow with one task per ``(env, seed)`` row from
the curated env YAMLs. Bundles return as datasets and are placed under
``odin_runs/<dispatch_id>/<run_id>/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import (
    DispatchState,
    SCHEMA_VERSION,
    write_dispatch_state,
)
from tools.odin.bifrost.config import load_bifrost_config
from tools.odin.bifrost.workflow import (
    RenderRow,
    osmo_safe_task_name,
    render_workflow_yaml,
    stage_source_tarball,
)


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odin-bifrost-dispatch",
        description="Dispatch Odin eval jobs to OSMO as a single workflow with N parallel tasks.",
    )
    p.add_argument("--osmo-config", required=True, type=Path, help="Path to bifrost-osmo.yaml.")
    p.add_argument("--physx-yaml", required=True, type=Path, help="Curated physx env list.")
    p.add_argument("--newton-yaml", type=Path, default=None, help="Curated newton env list (optional).")
    p.add_argument("--seeds", required=True, type=_parse_seeds, help="Comma-separated seeds.")
    p.add_argument("--include", type=str, default=None, help="Glob filter on task_id.")
    p.add_argument("--pool", type=str, default=None, help="Override config.pool.")
    p.add_argument("--priority", choices=["HIGH", "NORMAL", "LOW"], default=None)
    p.add_argument("--rsync", action="store_true", help="Enable continuous rsync of source_root for dev.")
    p.add_argument("--dry-run", action="store_true", help="Render workflow YAML and exit; do not submit.")
    p.add_argument("--resume", type=str, default=None, help="<dispatch_id> | LATEST")
    p.add_argument("--retry-failed", type=str, default=None, help="Comma-separated run_ids.")
    p.add_argument("--poll-interval", type=int, default=15, help="Seconds between OSMO status polls.")
    p.add_argument("--runs-root", type=Path, default=Path("./odin_runs"))
    p.add_argument("--verbose", action="store_true")
    return p


@dataclass(frozen=True)
class _PlannedRow:
    run_id: str
    task_id: str
    framework: str  # rsl-rl | skrl
    backend: str  # physx | newton
    seed: int
    num_envs: int
    max_iterations: int


def _load_envs_yaml(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return [e for e in (data.get("envs") or []) if e.get("keep") is True]


def _matches_include(task_id: str, include_glob: str | None) -> bool:
    if not include_glob:
        return True
    import fnmatch

    return fnmatch.fnmatch(task_id, include_glob)


def _build_rows(
    *,
    physx_yaml: Path,
    newton_yaml: Path | None,
    seeds: list[int],
    include_glob: str | None,
    dispatch_id: str,
) -> list[_PlannedRow]:
    rows: list[_PlannedRow] = []
    for path, backend in [(physx_yaml, "physx"), (newton_yaml, "newton")]:
        if path is None:
            continue
        for env in _load_envs_yaml(path):
            task_id = str(env["task_id"])
            if not _matches_include(task_id, include_glob):
                continue
            framework = str(env["framework"])
            num_envs = int(env["num_envs"])
            max_iter = int(env["max_iterations"])
            for seed in seeds:
                run_id = f"{framework}_{backend}_{task_id}_{dispatch_id}_seed{seed}"
                rows.append(
                    _PlannedRow(
                        run_id=run_id,
                        task_id=task_id,
                        framework=framework,
                        backend=backend,
                        seed=seed,
                        num_envs=num_envs,
                        max_iterations=max_iter,
                    )
                )
    return rows


def _planned_to_render(row: _PlannedRow) -> RenderRow:
    return RenderRow(
        run_id=row.run_id,
        osmo_task_name=osmo_safe_task_name(row.run_id),
        framework=row.framework,
        framework_runner="hugin" if row.framework == "rsl-rl" else "munin",
        task_id=row.task_id,
        backend=row.backend,
        seed=row.seed,
        num_envs=row.num_envs,
        max_iterations=row.max_iterations,
    )


def _planned_to_job(row: _PlannedRow) -> JobEntry:
    return JobEntry(
        run_id=row.run_id,
        task_id=row.task_id,
        framework=row.framework,
        backend=row.backend,
        num_envs=row.num_envs,
        max_iterations=row.max_iterations,
        seed=row.seed,
        bundle_dir_name=row.run_id,
        status="pending",
        osmo_task_name=osmo_safe_task_name(row.run_id),
    )


def _allocate_dispatch_id(now: dt.datetime | None = None) -> str:
    return (now or dt.datetime.utcnow()).strftime("%Y%m%d-%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_bifrost_config(args.osmo_config)
    if args.pool:
        cfg = _replace(cfg, pool=args.pool)
    if args.priority:
        cfg = _replace(cfg, priority=args.priority)

    dispatch_id = _allocate_dispatch_id()
    dispatch_dir = args.runs_root / dispatch_id
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(
        physx_yaml=args.physx_yaml,
        newton_yaml=args.newton_yaml,
        seeds=args.seeds,
        include_glob=args.include,
        dispatch_id=dispatch_id,
    )
    if not rows:
        print("No keep:true rows matched the include filter.", file=sys.stderr)
        return 2

    tarball_path: str | None = None
    if cfg.code_delivery.mode == "files_upload":
        tarball_path_p = dispatch_dir / "odin-source.tar.gz"
        repo_root = Path.cwd()
        stage_source_tarball(repo_root / cfg.code_delivery.source_root, tarball_path_p, repo_root=repo_root)
        tarball_path = str(tarball_path_p)

    workflow_yaml = render_workflow_yaml(
        dispatch_id=dispatch_id,
        rows=[_planned_to_render(r) for r in rows],
        cfg=cfg,
        tarball_path=tarball_path,
    )
    workflow_yaml_path = dispatch_dir / "workflow.yaml"
    workflow_yaml_path.write_text(workflow_yaml)

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id=dispatch_id,
        started_at=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        ended_at=None,
        seeds=list(args.seeds),
        commit_sha="",
        fleet=[],
        jobs=[_planned_to_job(r) for r in rows],
        dispatcher="osmo",
        osmo_workflow_id=None,
    )
    write_dispatch_state(dispatch_dir, state)

    if args.dry_run:
        print(f"[dry-run] wrote {workflow_yaml_path}")
        return 0

    # Submit + poll path lands in the next task.
    print("submission path not yet implemented; use --dry-run for now", file=sys.stderr)
    return 1


def _replace(cfg, **changes):
    """Lightweight dataclasses.replace, kept inline to avoid an import."""
    from dataclasses import replace

    return replace(cfg, **changes)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Create example config**

`tools/odin/config/bifrost-osmo.yaml.example`:

```yaml
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference bifrost-osmo.yaml. Copy to bifrost-osmo.yaml and edit.
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL

image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
  pull_credential: ngc-readonly

defaults:
  resources:
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000
  exec_timeout: 14400
  queue_timeout: 7200

retry:
  reschedule_codes: "3001-3006"
  restart_codes: ""

bundle_dataset_prefix: odin

code_delivery:
  mode: files_upload
  source_root: tools/odin
```

- [ ] **Step 6: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py -v --confcutdir=tools/odin
```

Expected: PASS (both dry-run tests).

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/cli.py tools/odin/tests/test_bifrost_cli.py tools/odin/config/bifrost-osmo.yaml.example
git commit -m "bifrost: add odin-bifrost-dispatch CLI with --dry-run"
```

---

### Task 14: Wire up real submission + polling in CLI

**Files:**
- Modify: `tools/odin/bifrost/cli.py`
- Modify: `tools/odin/tests/test_bifrost_cli.py`

- [ ] **Step 1: Write failing test for real submission flow (with mocked client)**

Append to `tools/odin/tests/test_bifrost_cli.py`:

```python
def test_main_submits_and_polls(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
    from tools.odin.asgard.state import read_dispatch_state
    from tools.odin.bifrost import cli as bifrost_cli
    from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot

    runs_root = tmp_path / "odin_runs"

    class FakeClient:
        def __init__(self):
            self.submit_calls: list[Path] = []
            self.status_calls = 0
            self.download_calls: list[tuple[str, Path]] = []

        def submit(self, yaml_path, *, rsync_pairs=()):
            self.submit_calls.append(yaml_path)
            return "wf-test-1"

        def status(self, workflow_id):
            self.status_calls += 1
            # Single keep:true env × 1 seed = 1 task; first poll RUNNING, second COMPLETED.
            if self.status_calls == 1:
                return WorkflowSnapshot("wf-test-1", "RUNNING", [TaskSnapshot("X", "RUNNING", None)])
            return WorkflowSnapshot("wf-test-1", "COMPLETED", [TaskSnapshot("X", "COMPLETED", 0)])

        def dataset_download(self, name, dest_dir):
            self.download_calls.append((name, dest_dir))
            # Synthesize a minimal valid bundle.
            run_dir = Path(dest_dir) / "rsl-rl_physx_Isaac-Ant-Direct-v0_*"
            # Caller passes the actual run_id-prefixed dataset name; we just
            # write a manifest into the first job's bundle_dir_name folder.

    fake = FakeClient()

    def fake_validator(p):
        return True

    # We need bifrost.cli to use FakeClient + a known osmo_task_name.
    with patch("tools.odin.bifrost.cli.OsmoClient", return_value=fake), \
         patch("tools.odin.bifrost.cli._manifest_validator", return_value=fake_validator):
        rc = bifrost_cli.main(
            [
                "--osmo-config",
                str(example_config),
                "--physx-yaml",
                str(example_physx_yaml),
                "--seeds",
                "42",
                "--runs-root",
                str(runs_root),
                "--poll-interval",
                "0",
            ]
        )
    assert rc == 0
    assert fake.submit_calls
    assert fake.status_calls >= 2
    dispatch_dir = list(runs_root.iterdir())[0]
    state = read_dispatch_state(dispatch_dir)
    assert state.osmo_workflow_id == "wf-test-1"
    assert state.jobs[0].status == "completed"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py::test_main_submits_and_polls -v --confcutdir=tools/odin
```

Expected: FAIL — submission returns 1 today.

- [ ] **Step 3: Replace the stubbed submission tail with real flow**

In `tools/odin/bifrost/cli.py`, replace the section after `if args.dry_run:` with:

```python
    if args.dry_run:
        print(f"[dry-run] wrote {workflow_yaml_path}")
        return 0

    client = OsmoClient(profile=cfg.osmo_profile)
    rsync_pairs: list[tuple[str, str]] = []
    if args.rsync:
        rsync_pairs.append((cfg.code_delivery.source_root, "/workspace/IsaacLab/" + cfg.code_delivery.source_root))
    workflow_id = client.submit(workflow_yaml_path, rsync_pairs=rsync_pairs)
    state.osmo_workflow_id = workflow_id
    write_dispatch_state(dispatch_dir, state)

    validator = _manifest_validator(dispatch_dir)

    def on_completed(job: JobEntry) -> None:
        dataset_name = f"{cfg.bundle_dataset_prefix}-{dispatch_id}-{job.run_id}"
        download_and_validate_bundle(
            client=client,
            dataset_name=dataset_name,
            dispatch_dir=dispatch_dir,
            run_id=job.run_id,
            validator=validator,
        )

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=dispatch_dir,
        on_task_completed=on_completed,
        poll_interval_s=float(args.poll_interval),
    )
    state.ended_at = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    write_dispatch_state(dispatch_dir, state)
    return 0


def _manifest_validator(dispatch_dir):
    """Return a callable that validates a bundle directory's manifest.

    Stub implementation: a manifest is "valid" if the file exists. Replace
    with the canonical validator from ``tools.odin.common.manifest`` when
    that exposes a public ``validate(path) -> bool`` API.
    """
    def _validate(bundle_dir):
        return (bundle_dir / "manifest.json").exists()
    return _validate
```

Add the imports at the top:

```python
from tools.odin.bifrost.bundle import download_and_validate_bundle
from tools.odin.bifrost.client import OsmoClient
from tools.odin.bifrost.poller import poll_until_terminal
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py -v --confcutdir=tools/odin
```

Expected: PASS, all three CLI tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/cli.py tools/odin/tests/test_bifrost_cli.py
git commit -m "bifrost: wire up real OSMO submission + polling in CLI"
```

---

### Task 15: Add `--resume` support

**Files:**
- Modify: `tools/odin/bifrost/cli.py`
- Modify: `tools/odin/tests/test_bifrost_cli.py`

- [ ] **Step 1: Write failing test**

Append to `tools/odin/tests/test_bifrost_cli.py`:

```python
def test_resume_reattaches_to_existing_dispatch(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
    """--resume LATEST should NOT create a new dispatch dir; it reuses the existing one."""
    from tools.odin.asgard.state import read_dispatch_state
    from tools.odin.bifrost import cli as bifrost_cli
    from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot

    runs_root = tmp_path / "odin_runs"

    # First, do a dry-run to create a dispatch dir + state.
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(example_config),
            "--physx-yaml",
            str(example_physx_yaml),
            "--seeds",
            "42",
            "--runs-root",
            str(runs_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    [dispatch_dir] = list(runs_root.iterdir())
    # Pretend we'd submitted: stamp a workflow id.
    state = read_dispatch_state(dispatch_dir)
    state.osmo_workflow_id = "wf-already-running"
    from tools.odin.asgard.state import write_dispatch_state

    write_dispatch_state(dispatch_dir, state)

    class FakeClient:
        def submit(self, *a, **k):
            raise AssertionError("resume must NOT call submit")

        def status(self, wf):
            assert wf == "wf-already-running"
            return WorkflowSnapshot(wf, "COMPLETED", [TaskSnapshot(state.jobs[0].osmo_task_name, "COMPLETED", 0)])

        def dataset_download(self, name, dest):
            (dest / state.jobs[0].run_id).mkdir(parents=True, exist_ok=True)
            (dest / state.jobs[0].run_id / "manifest.json").write_text("{}")

    with patch("tools.odin.bifrost.cli.OsmoClient", return_value=FakeClient()):
        rc = bifrost_cli.main(
            [
                "--osmo-config",
                str(example_config),
                "--physx-yaml",
                str(example_physx_yaml),
                "--seeds",
                "42",
                "--runs-root",
                str(runs_root),
                "--poll-interval",
                "0",
                "--resume",
                "LATEST",
            ]
        )
    assert rc == 0
    # No new dispatch dir was created.
    assert len(list(runs_root.iterdir())) == 1
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py::test_resume_reattaches_to_existing_dispatch -v --confcutdir=tools/odin
```

Expected: FAIL — current `main` always creates a new dispatch dir.

- [ ] **Step 3: Add resume logic in `main`**

In `tools/odin/bifrost/cli.py`, before the "allocate dispatch_id" block:

```python
    if args.resume:
        dispatch_dir = _resolve_resume_dispatch(args.runs_root, args.resume)
        state = read_dispatch_state(dispatch_dir)
        if state is None:
            print(f"resume target {dispatch_dir} has no dispatch.json", file=sys.stderr)
            return 2
        if state.osmo_workflow_id is None:
            print(f"resume target {dispatch_dir} has no osmo_workflow_id (was --dry-run only?)", file=sys.stderr)
            return 2
        cfg = load_bifrost_config(args.osmo_config)
        client = OsmoClient(profile=cfg.osmo_profile)
        validator = _manifest_validator(dispatch_dir)

        def on_completed(job: JobEntry) -> None:
            dataset_name = f"{cfg.bundle_dataset_prefix}-{state.dispatch_id}-{job.run_id}"
            download_and_validate_bundle(
                client=client,
                dataset_name=dataset_name,
                dispatch_dir=dispatch_dir,
                run_id=job.run_id,
                validator=validator,
            )

        poll_until_terminal(
            client=client,
            state=state,
            dispatch_dir=dispatch_dir,
            on_task_completed=on_completed,
            poll_interval_s=float(args.poll_interval),
        )
        state.ended_at = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        write_dispatch_state(dispatch_dir, state)
        return 0
```

And add the resolver helper:

```python
def _resolve_resume_dispatch(runs_root: Path, target: str) -> Path:
    if target == "LATEST":
        candidates = sorted([p for p in runs_root.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"no dispatch dirs under {runs_root}")
        return candidates[-1]
    return runs_root / target
```

Add the import at the top:

```python
from tools.odin.asgard.state import read_dispatch_state
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py -v --confcutdir=tools/odin
```

Expected: PASS, all four CLI tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/cli.py tools/odin/tests/test_bifrost_cli.py
git commit -m "bifrost: add --resume to re-attach to in-flight workflow"
```

---

### Task 16: Add `--retry-failed` support

**Files:**
- Modify: `tools/odin/bifrost/cli.py`
- Modify: `tools/odin/tests/test_bifrost_cli.py`

- [ ] **Step 1: Write failing test**

Append to `tools/odin/tests/test_bifrost_cli.py`:

```python
def test_retry_failed_creates_child_dispatch_with_only_failed_rows(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
    from tools.odin.asgard.jobs import FailureInfo
    from tools.odin.asgard.state import read_dispatch_state, write_dispatch_state
    from tools.odin.bifrost import cli as bifrost_cli

    runs_root = tmp_path / "odin_runs"

    # Stand up a parent dispatch via dry-run, then mark its row as failed.
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(example_config),
            "--physx-yaml",
            str(example_physx_yaml),
            "--seeds",
            "42,43",
            "--runs-root",
            str(runs_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    [parent_dir] = list(runs_root.iterdir())
    parent = read_dispatch_state(parent_dir)
    parent.jobs[0].status = "failed"
    parent.jobs[0].failure = FailureInfo(kind="hugin_crash", message="boom", details={})
    parent.jobs[1].status = "completed"
    write_dispatch_state(parent_dir, parent)

    failed_run_id = parent.jobs[0].run_id
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(example_config),
            "--physx-yaml",
            str(example_physx_yaml),
            "--seeds",
            "42,43",
            "--runs-root",
            str(runs_root),
            "--retry-failed",
            failed_run_id,
            "--dry-run",
        ]
    )
    assert rc == 0
    dispatch_dirs = sorted(runs_root.iterdir())
    assert len(dispatch_dirs) == 2
    child_dir = dispatch_dirs[-1]
    child = read_dispatch_state(child_dir)
    assert child.parent_dispatch_id == parent.dispatch_id
    assert len(child.jobs) == 1
    assert child.jobs[0].run_id == failed_run_id
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py::test_retry_failed_creates_child_dispatch_with_only_failed_rows -v --confcutdir=tools/odin
```

Expected: FAIL — `--retry-failed` is currently parsed but ignored.

- [ ] **Step 3: Implement retry-failed flow**

After `_build_rows(...)` and before `if not rows:`, add:

```python
    parent_dispatch_id: str | None = None
    if args.retry_failed:
        retry_run_ids = {x.strip() for x in args.retry_failed.split(",") if x.strip()}
        # Locate the most recent dispatch that has these run_ids.
        parent = _find_parent_dispatch(args.runs_root, retry_run_ids)
        if parent is None:
            print(f"no recent dispatch contains all run_ids {sorted(retry_run_ids)}", file=sys.stderr)
            return 2
        parent_dispatch_id = parent.dispatch_id
        rows = [r for r in rows if r.run_id in retry_run_ids]
        if not rows:
            print(
                "retry-failed run_ids did not match any rows produced by current "
                "physx/newton/seeds args; pass the same args you used for the parent",
                file=sys.stderr,
            )
            return 2
```

Then, when constructing `DispatchState`, plumb `parent_dispatch_id`:

```python
    state = DispatchState(
        # ... existing kwargs ...
        dispatcher="osmo",
        osmo_workflow_id=None,
        parent_dispatch_id=parent_dispatch_id,
    )
```

Add the helper:

```python
def _find_parent_dispatch(runs_root: Path, retry_run_ids: set[str]) -> DispatchState | None:
    if not runs_root.exists():
        return None
    for d in sorted(runs_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        st = read_dispatch_state(d)
        if st is None:
            continue
        present = {j.run_id for j in st.jobs}
        if retry_run_ids <= present:
            return st
    return None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py -v --confcutdir=tools/odin
```

Expected: PASS, all five CLI tests green.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/cli.py tools/odin/tests/test_bifrost_cli.py
git commit -m "bifrost: add --retry-failed to spawn child dispatch with failed rows only"
```

---

## Phase 4 — Polish

### Task 17: Add `--verbose` live log tail

**Files:**
- Modify: `tools/odin/bifrost/cli.py`
- Modify: `tools/odin/tests/test_bifrost_cli.py`

A small thread that picks the first RUNNING task and tees its logs to disk. Simple, opt-in, single-task.

- [ ] **Step 1: Write failing test**

Append to `tools/odin/tests/test_bifrost_cli.py`:

```python
def test_verbose_tail_writes_log_file(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
    from tools.odin.bifrost import cli as bifrost_cli
    from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot

    runs_root = tmp_path / "odin_runs"

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def submit(self, *a, **k):
            return "wf-1"

        def status(self, wf):
            self.calls += 1
            if self.calls == 1:
                # One task RUNNING — tail should attach.
                return WorkflowSnapshot(wf, "RUNNING", [TaskSnapshot("X", "RUNNING", None)])
            return WorkflowSnapshot(wf, "COMPLETED", [TaskSnapshot("X", "COMPLETED", 0)])

        def logs(self, wf, task, *, follow):
            yield b"hello from osmo\n"

        def dataset_download(self, name, dest):
            run = list(read_dispatch_state(dest).jobs)[0].run_id
            (dest / run).mkdir(parents=True, exist_ok=True)
            (dest / run / "manifest.json").write_text("{}")

    from tools.odin.asgard.state import read_dispatch_state

    with patch("tools.odin.bifrost.cli.OsmoClient", return_value=FakeClient()):
        rc = bifrost_cli.main(
            [
                "--osmo-config",
                str(example_config),
                "--physx-yaml",
                str(example_physx_yaml),
                "--seeds",
                "42",
                "--runs-root",
                str(runs_root),
                "--poll-interval",
                "0",
                "--verbose",
            ]
        )
    assert rc == 0
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    log_path = dispatch_dir / state.jobs[0].run_id / "logs" / "osmo-tail.log"
    assert log_path.exists()
    assert b"hello from osmo" in log_path.read_bytes()
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py::test_verbose_tail_writes_log_file -v --confcutdir=tools/odin
```

Expected: FAIL — no log tail mechanism yet.

- [ ] **Step 3: Add a one-task tail thread**

In `tools/odin/bifrost/cli.py`, just before the `poll_until_terminal` call in the non-resume path:

```python
    tail_stop = threading.Event()
    tail_thread: threading.Thread | None = None
    if args.verbose:
        tail_thread = threading.Thread(
            target=_tail_first_running_task,
            args=(client, state, dispatch_dir, tail_stop),
            daemon=True,
        )
        tail_thread.start()
```

After `poll_until_terminal` returns:

```python
    tail_stop.set()
    if tail_thread is not None:
        tail_thread.join(timeout=5)
```

Add the function:

```python
def _tail_first_running_task(
    client,
    state: DispatchState,
    dispatch_dir: Path,
    stop_event: threading.Event,
) -> None:
    """Best-effort live tail of the first task we observe RUNNING.

    Single-task by design (per spec §6.1 step 8). When that task terminates,
    we don't pick up another — the next dispatch's --verbose will.
    """
    while not stop_event.is_set():
        try:
            snap = client.status(state.osmo_workflow_id)
        except Exception:
            time.sleep(2)
            continue
        running = [t for t in snap.tasks if t.status == "RUNNING"]
        if running:
            target = running[0]
            job = next((j for j in state.jobs if j.osmo_task_name == target.name), None)
            if job is None:
                return
            log_dir = dispatch_dir / job.run_id / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "osmo-tail.log"
            with log_path.open("ab") as fh:
                for chunk in client.logs(state.osmo_workflow_id, target.name, follow=True):
                    if stop_event.is_set():
                        return
                    fh.write(chunk)
                    fh.flush()
            return
        if stop_event.wait(2):
            return
```

Add imports:

```python
import threading
import time
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_cli.py -v --confcutdir=tools/odin
```

Expected: PASS, all six CLI tests green. Note: the verbose tail test is timing-sensitive on the `daemon` thread; if it's flaky, the test can `time.sleep(0.2)` between `submit` and the assertion to give the tail thread a chance to write.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/bifrost/cli.py tools/odin/tests/test_bifrost_cli.py
git commit -m "bifrost: add --verbose live log tail for first RUNNING task"
```

---

### Task 18: Update README

**Files:**
- Modify: `tools/odin/README.md`

- [ ] **Step 1: Read the existing README sections to figure out where to splice**

```bash
grep -n '^##\|^###' tools/odin/README.md
```

- [ ] **Step 2: Add a "Dispatching to OSMO (Bifrost)" section after the asgard dispatch section**

In `tools/odin/README.md`, after the "Dispatching across a fleet (T3.1 — Asgard)" section and before "Aggregating a dispatch (T4.1 — Valhalla)", add:

```markdown
## Dispatching to OSMO (Bifrost)

`tools/odin/bifrost/cli.py` (the `odin-bifrost-dispatch` entry point) is the
peer of `odin-dispatch` for sites where the compute is managed by
[OSMO](https://github.com/NVIDIA/OSMO). Bifrost submits a single OSMO
workflow with N parallel tasks (one per `(env, seed)` row); bundles return
as datasets and land at `odin_runs/<dispatch_id>/<run_id>/` — the same
layout that asgard produces. Valhalla aggregation is unchanged.

There is no fleet config: OSMO owns scheduling, image pull, infrastructure
retry (`exitActions`), and output upload. A bifrost-osmo.yaml only
captures *what to ask OSMO for*. Copy
`tools/odin/config/bifrost-osmo.yaml.example` and edit:

```yaml
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL
image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
  pull_credential: ngc-readonly
defaults:
  resources: {cpu: 16, gpu: 1, memory: 64Gi, storage: 64Gi, platform: rtx-pro-6000}
  exec_timeout: 14400
  queue_timeout: 7200
retry:
  reschedule_codes: "3001-3006"
  restart_codes: ""
bundle_dataset_prefix: odin
code_delivery:
  mode: files_upload    # files_upload | rsync | image_baked
  source_root: tools/odin
```

### Running a bifrost dispatch

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/bifrost/cli.py \
    --osmo-config bifrost-osmo.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    [--newton-yaml tools/odin/config/newton_envs.yaml] \
    --seeds 42,43,44 \
    [--include 'Isaac-Ant-*'] \
    [--pool rtx-pro-6000-eval] \
    [--priority HIGH|NORMAL|LOW] \
    [--rsync] \
    [--dry-run] \
    [--resume <dispatch_id|LATEST>] \
    [--retry-failed <run_ids>] \
    [--poll-interval 15] \
    [--verbose]
```

`--dry-run` renders the workflow YAML and writes `dispatch.json` without
submitting — handy for inspecting what bifrost would send to OSMO.

### Failure handling

OSMO terminal task states map to Odin's four-kind failure classification
in `tools/odin/bifrost/poller.py::OSMO_STATE_TO_FAILURE_KIND`:

| OSMO state                                                       | Odin `failure.kind`        |
|---|---|
| `COMPLETED`                                                      | (success)                  |
| `FAILED`                                                         | `hugin_crash`              |
| `FAILED_EXEC_TIMEOUT`                                            | `timeout`                  |
| `FAILED_BACKEND_ERROR`, `FAILED_PREEMPTED`, `FAILED_EVICTED`,    | `infrastructure`           |
| `FAILED_IMAGE_PULL`, `FAILED_START_*`, `FAILED_QUEUE_TIMEOUT`,   |                            |
| `FAILED_SERVER_ERROR`, `FAILED_CANCELED`                         |                            |
| `COMPLETED` with missing/malformed manifest                      | `hugin_malformed_bundle`   |

OSMO automatically reschedules `FAILED_BACKEND_ERROR` and friends per
the `retry.reschedule_codes` range in the config — bifrost only reports
the kind once OSMO gives up.

User-class retries (`hugin_crash`, `timeout`, `hugin_malformed_bundle`)
are explicit operator action via `--retry-failed`, which submits a new
workflow with only the named rows and links the new dispatch back via
`parent_dispatch_id`.

### State on disk

```
odin_runs/
└── 20260505-150000/
    ├── dispatch.json           # schema 1.5; dispatcher: "osmo"; osmo_workflow_id: ...
    ├── workflow.yaml           # the rendered OSMO workflow
    ├── odin-source.tar.gz      # uploaded with files_upload mode
    └── <run_id>/
        ├── manifest.json
        ├── training.json
        ├── startup.json
        ├── training_data/
        └── logs/
            └── osmo-tail.log   # only if --verbose
```
```

- [ ] **Step 3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/README.md
git commit -m "docs: add bifrost (OSMO) dispatch section to tools/odin/README"
```

---

### Task 19: Slow-marked integration test (gated)

**Files:**
- Create: `tools/odin/tests/test_bifrost_integration.py`

This test only runs when `ODIN_OSMO_INTEGRATION=1` and a local OSMO is reachable. It's a smoke test, not a coverage net.

- [ ] **Step 1: Confirm pytest's `slow` marker is configured**

```bash
grep -rn 'markers\s*=\|"slow"' tools/odin/ pyproject.toml setup.cfg pytest.ini 2>/dev/null | head -5
```

If a `slow` marker isn't already registered (it likely is, given the existing `test_asgard_integration.py`), add it to the right config file. If unsure, replicate the exact `pytestmark` line that the existing asgard integration test uses.

- [ ] **Step 2: Create the integration test**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke test against a local OSMO deployment.

Gated by ``ODIN_OSMO_INTEGRATION=1``. Skips otherwise.
Optimistic path only: render → submit → poll → download → assert layout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow]


def _osmo_available() -> bool:
    if os.environ.get("ODIN_OSMO_INTEGRATION") != "1":
        return False
    if shutil.which("osmo") is None:
        return False
    cp = subprocess.run(["osmo", "profile", "list"], capture_output=True, text=True)
    return cp.returncode == 0


@pytest.mark.skipif(not _osmo_available(), reason="ODIN_OSMO_INTEGRATION!=1 or osmo CLI unavailable")
def test_bifrost_end_to_end_smoke(tmp_path: Path):
    """Submit a 1-task workflow that writes a fake manifest, then verify the bundle lands on disk."""
    from tools.odin.bifrost import cli as bifrost_cli

    cfg = tmp_path / "bifrost-osmo.yaml"
    cfg.write_text(
        # Minimal config pointing at a tiny image. Adjust pool/platform per local OSMO.
        "osmo_profile: " + os.environ.get("ODIN_OSMO_PROFILE", "default") + "\n"
        "pool: " + os.environ.get("ODIN_OSMO_POOL", "default") + "\n"
        "priority: NORMAL\n"
        "image:\n"
        "  reference: alpine:3.18\n"
        "defaults:\n"
        "  resources: {cpu: 1, gpu: 0, memory: 256Mi, storage: 256Mi, platform: cpu}\n"
        "  exec_timeout: 60\n"
        "  queue_timeout: 120\n"
        "retry: {reschedule_codes: '', restart_codes: ''}\n"
        "bundle_dataset_prefix: odin-int-test\n"
        "code_delivery: {mode: image_baked, source_root: tools/odin}\n"
    )
    physx = tmp_path / "physx.yaml"
    physx.write_text(
        "envs:\n"
        "- task_id: smoke-task\n"
        "  framework: rsl-rl\n"
        "  num_envs: 1\n"
        "  max_iterations: 1\n"
        "  keep: true\n"
    )

    runs_root = tmp_path / "odin_runs"
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(cfg),
            "--physx-yaml",
            str(physx),
            "--seeds",
            "1",
            "--runs-root",
            str(runs_root),
            "--poll-interval",
            "5",
        ]
    )
    assert rc == 0
    [dispatch_dir] = list(runs_root.iterdir())
    state_path = dispatch_dir / "dispatch.json"
    state = json.loads(state_path.read_text())
    assert state["dispatcher"] == "osmo"
    # Note: this test will fail at runtime because the alpine image can't run hugin.
    # When you wire a real test image with a stub manifest writer, replace the
    # `image_baked` config with the real reference and assert manifest.json exists.
```

- [ ] **Step 3: Run the test gated (expect skip)**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_integration.py -v --confcutdir=tools/odin
```

Expected: SKIPPED (no `ODIN_OSMO_INTEGRATION=1` set).

- [ ] **Step 4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_bifrost_integration.py
git commit -m "bifrost: add slow-marked integration test (env-gated)"
```

---

## Final verification

### Task 20: Run the full test suite + cleanup

- [ ] **Step 1: Run all bifrost tests (fast)**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_*.py -v --confcutdir=tools/odin -m "not slow"
```

Expected: PASS — every fast bifrost test green.

- [ ] **Step 2: Run all asgard tests to confirm no regressions**

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/ -v --confcutdir=tools/odin -m "not slow"
```

Expected: PASS — including `test_asgard_state.py` with the new schema 1.5 expectations.

- [ ] **Step 3: Final pre-commit pass**

```bash
./isaaclab.sh -f
```

Expected: clean (no modifications). If anything was modified, stage and commit those tweaks separately with a "style: pre-commit cleanup" message.

- [ ] **Step 4: Confirm git log**

```bash
git log --oneline antoiner/feat/odin -20
```

You should see the bifrost commits in roughly the order of these tasks. If any task got squashed or reordered during execution, that's fine as long as each landed test was green when committed.

---

## Self-Review Notes

This plan was self-reviewed against the spec on 2026-05-05. Items confirmed:

- **Spec coverage:** Every spec section is implemented by at least one task. §3 (context) is informational. §4 module layout → Tasks 2–6, 7–9, 10–12, 13. §5.1 config → Task 3. §5.2 CLI → Tasks 13–17. §5.3 dispatch.json → Task 1. §6.1 lifecycle → Tasks 13–14. §6.2 resume → Task 15. §6.3 retry-failed → Task 16. §7 failure mapping → Task 10. §8 retry semantics → Tasks 10, 16. §9 internals → Tasks 4–12. §10 testing → covered by per-task tests + Task 19 integration. §11 naming → Task 2 docstring + Task 18 README.
- **Placeholder scan:** No "TBD", no "implement later", no "similar to Task N". Every code step has the actual code.
- **Type consistency:** `RenderRow`, `BifrostConfig`, `WorkflowSnapshot`, `TaskSnapshot`, `JobEntry`, `DispatchState`, `BundleResult`, `OSMO_STATE_TO_FAILURE_KIND` — every name used in a later task is defined in an earlier task. Method signatures consistent across tests and implementations.
- **Out of scope checks:** Mixed-backend, multi-node, custom image, dashboard link, asgard rename — none included, per spec.
