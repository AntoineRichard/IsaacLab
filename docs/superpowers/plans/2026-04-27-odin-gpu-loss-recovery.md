# Odin GPU-Loss Detection + Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect GPU loss before/during/after each Odin job; auto-recover the host's container when possible; fail jobs out cleanly to another host when not.

**Architecture:** Four cooperating layers — Hugin/Munin verify output JSON exists (not just returncode); preflight adds a `gpu_present` check; worker classifier emits new `failure_kind="gpu_lost"` when stderr matches NVML / CUDA / Vulkan signatures; worker auto-runs a container-restart recovery tool, retries on same host on success, marks host down on failure.

**Tech Stack:** Python 3.10+, pytest, dataclasses, existing `tools.odin.asgard.transport.SSHRunner` interface (no new pip deps).

**Branch:** `antoiner/feat/odin` (continues from spec commit `192735a8d83`).

**Spec:** `docs/superpowers/specs/2026-04-27-odin-gpu-loss-recovery-design.md`.

---

## Conventions used in every task

- **Pre-commit:** Run `./isaaclab.sh -f` BEFORE `git commit`. If it modifies files, `git add` them and rerun until clean.
- **Pytest invocation:** `./isaaclab.sh -p -m pytest <path> -v`. Single test: append `::test_name`.
- **Commit message:** Imperative subject ≤ 50 chars; body wrapped at 72 chars; **NO** AI co-authorship lines (per `AGENTS.md`).
- **Run all Odin tests once after each task:** `./isaaclab.sh -p -m pytest tools/odin/tests -q`. Should stay green throughout.
- **No new dependencies.** All work uses stdlib + existing `tools.odin.asgard.transport.SSHRunner`.

---

## File map — what gets created or modified

| File | Owner task | Responsibility |
|---|---|---|
| `tools/odin/asgard/recovery.py` | T1 | `RecoveryResult` dataclass + `recover_valkyrie_gpu(host, *, ssh)` function |
| `tools/odin/tests/test_recovery.py` | T1 | Recovery-tool unit tests (5 cases) |
| `tools/odin/asgard/recovery_cli.py` | T2 | `python -m tools.odin.asgard.recovery_cli --fleet … --host …` wrapper |
| `tools/odin/tests/test_recovery_cli.py` | T2 | CLI unit tests |
| `tools/odin/hugin/run.py` | T3 | `_run_phase` adds output-existence check |
| `tools/odin/tests/test_hugin.py` | T3 | New test cases for output-missing case |
| `tools/odin/munin/run.py` | T4 | Mirror of T3 in Munin |
| `tools/odin/tests/test_munin.py` | T4 | Mirror of T3 in Munin |
| `tools/odin/asgard/preflight.py` | T5 | Add 5th check `gpu_present` |
| `tools/odin/tests/test_asgard_preflight.py` | T5 | New test cases for `gpu_present` |
| `tools/odin/asgard/jobs.py` | T6 | `FailureInfo` docstring documents `gpu_lost` kind |
| `tools/odin/asgard/worker.py` (`_classify` only) | T7 | GPU-loss stderr matcher emitting `kind="gpu_lost"` |
| `tools/odin/tests/test_asgard_worker.py` | T7 | Classifier tests for 3 signatures + non-false-positive |
| `tools/odin/asgard/worker.py` (`_execute` recovery integration) | T8 | Recovery loop, host-down emission, `preferred_not` update |
| `tools/odin/tests/test_asgard_worker.py` | T8 | Recovery success/fail/three-in-a-row tests |
| `tools/odin/asgard/runner.py` | T9 | Consume `recovered` / `host_down` events; update `FleetSnapshot` |
| `tools/odin/tests/test_asgard_runner.py` | T9 | Unit tests for new transitions |
| `tools/odin/asgard/state.py` | T10 | `SCHEMA_VERSION = "1.3"`; resume-from-1.2 still works |
| `tools/odin/tests/test_asgard_state.py` | T10 | Schema-version test + 1.2-resume test |
| `tools/odin/tests/test_asgard_integration.py` | T11 | End-to-end slow-marked test for `gpu_lost` flow |
| `docs/odin/architecture.md` | T11 | Change-log entry + "Last updated" bump |

---

## Task 1: Recovery tool module

**Files:**
- Create: `tools/odin/asgard/recovery.py`
- Test: `tools/odin/tests/test_recovery.py`

This task delivers a stand-alone module the worker (T8) and CLI (T2) will both call. SSH-only; no docker/Python-on-host needed. All 5 recovery edge cases from the spec land here.

- [ ] **Step 1.1: Write the failing tests** — `tools/odin/tests/test_recovery.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.recovery.recover_valkyrie_gpu`."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.recovery import RecoveryResult, recover_valkyrie_gpu
from tools.odin.asgard.transport import SSHResult


@dataclass
class _ScriptedSSH:
    """Replays a list of SSHResult in call order; raises if exhausted.

    Each call records the (host, cmd) pair into ``calls`` for assertion.
    """

    responses: list  # list[SSHResult]
    calls: list = None  # list[tuple[str, str]]

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        self.calls.append((host.host, cmd))
        if not self.responses:
            return SSHResult(exit_code=255, stdout="", stderr="ssh script exhausted", duration_s=0.0)
        return self.responses.pop(0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="horde", container_name="isaac-lab-base")


def _ok(stdout: str = "ok\n") -> SSHResult:
    return SSHResult(exit_code=0, stdout=stdout, stderr="", duration_s=0.01)


def _fail(stderr: str, exit_code: int = 1) -> SSHResult:
    return SSHResult(exit_code=exit_code, stdout="", stderr=stderr, duration_s=0.01)


def test_recovery_happy_path():
    ssh = _ScriptedSSH(
        responses=[
            _ok("isaac-lab-base\n"),                         # docker restart
            _ok("running\n"),                                # docker inspect (1st poll)
            _ok("GPU 0: NVIDIA A100 (UUID: GPU-abc...)\n"),  # nvidia-smi -L
        ]
    )
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert isinstance(r, RecoveryResult)
    assert r.attempted is True
    assert r.recovered is True
    assert r.host == "v1"
    assert r.container_name == "isaac-lab-base"
    assert "recovered_via_container_restart" in r.message
    assert r.details["docker_restart"] == "ok"
    assert r.details["gpu_probe"] == "ok"


def test_recovery_docker_restart_fails():
    ssh = _ScriptedSSH(responses=[_fail("Error response from daemon: container not running")])
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.attempted is True
    assert r.recovered is False
    assert r.message.startswith("docker_restart_failed")
    assert "docker_restart" in r.details
    # Subsequent phases not called.
    assert "container_up" not in r.details
    assert "gpu_probe" not in r.details


def test_recovery_container_never_running():
    # docker restart succeeds; inspect returns "created" forever.
    responses = [_ok("isaac-lab-base\n")] + [_ok("created\n")] * 20
    ssh = _ScriptedSSH(responses=responses)
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.recovered is False
    assert r.message == "container_not_running_after_restart"
    assert r.details["container_up"] == "timeout"
    assert "gpu_probe" not in r.details


def test_recovery_gpu_probe_empty():
    # docker restart ok, container running, but nvidia-smi -L returns empty stdout.
    ssh = _ScriptedSSH(
        responses=[
            _ok("isaac-lab-base\n"),
            _ok("running\n"),
            SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01),
        ]
    )
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.recovered is False
    assert r.message.startswith("gpu_probe_failed")


def test_recovery_ssh_unreachable():
    ssh = _ScriptedSSH(responses=[SSHResult(exit_code=255, stdout="", stderr="connection refused", duration_s=0.0)])
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    # ssh_unreachable is a special case: the FIRST SSH call (docker restart)
    # came back with exit 255, which we treat as an SSH-layer failure rather
    # than a docker-layer failure. attempted stays True (we tried), but the
    # detail differs from docker_restart_failed.
    assert r.recovered is False
    assert r.message == "ssh_unreachable"
```

- [ ] **Step 1.2: Run tests, verify they FAIL**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_recovery.py -v`
Expected: All five tests FAIL with `ModuleNotFoundError: No module named 'tools.odin.asgard.recovery'`.

- [ ] **Step 1.3: Implement `recovery.py`** — minimal code to make all 5 tests pass

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Container-level GPU-loss recovery for a single Valkyrie.

Used by :class:`tools.odin.asgard.worker.ValkyrieWorker` mid-dispatch and by
:mod:`tools.odin.asgard.recovery_cli` for ad-hoc operator recovery. Restarts
the named container, waits for it to reach ``running`` state, then probes
``nvidia-smi -L`` from inside it as the acceptance test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import SSHRunner

__all__ = ["RecoveryResult", "recover_valkyrie_gpu"]


_DOCKER_RESTART_TIMEOUT_S = 60.0
_CONTAINER_UP_POLL_INTERVAL_S = 2.0
_CONTAINER_UP_MAX_POLLS = 15  # 15 × 2 s = 30 s budget
_GPU_PROBE_TIMEOUT_S = 15.0


@dataclass
class RecoveryResult:
    """Outcome of one ``recover_valkyrie_gpu`` invocation."""

    host: str
    container_name: str
    attempted: bool
    recovered: bool
    duration_s: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def recover_valkyrie_gpu(host: ValkyrieConfig, *, ssh: SSHRunner) -> RecoveryResult:
    """Restart container, wait for ``running``, probe ``nvidia-smi -L``.

    Returns a :class:`RecoveryResult` whose ``recovered`` is True iff every
    phase succeeded. On failure, ``message`` is one of the well-known
    short strings used by the worker for telemetry.
    """
    started = time.monotonic()
    details: dict[str, Any] = {}

    # Phase 1: docker restart
    r = ssh.run(host, f"docker restart {host.container_name}", timeout_s=_DOCKER_RESTART_TIMEOUT_S)
    if r.exit_code == 255:
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message="ssh_unreachable",
            details={"docker_restart": "ssh_unreachable"},
        )
    if r.exit_code != 0:
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message=f"docker_restart_failed: {r.stderr.strip() or 'non-zero exit'}",
            details={"docker_restart": "failed", "stderr": r.stderr.strip()},
        )
    details["docker_restart"] = "ok"

    # Phase 2: poll docker inspect for State.Status == running
    inspect_cmd = f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}"
    container_running = False
    for _ in range(_CONTAINER_UP_MAX_POLLS):
        r = ssh.run(host, inspect_cmd, timeout_s=10.0)
        if r.exit_code == 0 and r.stdout.strip() == "running":
            container_running = True
            break
        time.sleep(_CONTAINER_UP_POLL_INTERVAL_S)
    if not container_running:
        details["container_up"] = "timeout"
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message="container_not_running_after_restart",
            details=details,
        )
    details["container_up"] = "ok"

    # Phase 3: GPU probe
    probe_cmd = f"docker exec {host.container_name} nvidia-smi -L"
    r = ssh.run(host, probe_cmd, timeout_s=_GPU_PROBE_TIMEOUT_S)
    if r.exit_code != 0 or not r.stdout.strip():
        details["gpu_probe"] = "failed"
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message=f"gpu_probe_failed: {r.stderr.strip() or 'empty stdout'}",
            details=details,
        )
    details["gpu_probe"] = "ok"
    details["gpu_probe_stdout"] = r.stdout.strip()

    return RecoveryResult(
        host=host.host,
        container_name=host.container_name,
        attempted=True,
        recovered=True,
        duration_s=time.monotonic() - started,
        message="recovered_via_container_restart",
        details=details,
    )
```

> **Note for the implementer:** the polling loop calls `time.sleep`, which would make `test_recovery_container_never_running` take 30 seconds in CI. Patch `time.sleep` to a no-op in tests via `monkeypatch.setattr("tools.odin.asgard.recovery.time.sleep", lambda _: None)`. Add this patch to that single test (Step 1.1 doesn't need it for the other four cases). Update the test:

```python
def test_recovery_container_never_running(monkeypatch):
    monkeypatch.setattr("tools.odin.asgard.recovery.time.sleep", lambda _: None)
    responses = [_ok("isaac-lab-base\n")] + [_ok("created\n")] * 20
    ssh = _ScriptedSSH(responses=responses)
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.recovered is False
    assert r.message == "container_not_running_after_restart"
    assert r.details["container_up"] == "timeout"
    assert "gpu_probe" not in r.details
```

- [ ] **Step 1.4: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_recovery.py -v`
Expected: 5/5 PASS.

- [ ] **Step 1.5: Run pre-commit**

Run: `./isaaclab.sh -f`
Expected: clean (or restage modified files and rerun until clean).

- [ ] **Step 1.6: Commit**

```bash
git add tools/odin/asgard/recovery.py tools/odin/tests/test_recovery.py
git commit -m "Add Odin recovery tool — container restart + GPU probe

New module tools/odin/asgard/recovery.py exposes
recover_valkyrie_gpu(host, *, ssh): SSH-driven docker restart followed
by a poll for State.Status == running and a nvidia-smi -L acceptance
probe. Returns RecoveryResult with phase-by-phase details. Used by the
worker's GPU-loss recovery loop (T8) and the odin-recover CLI (T2)."
```

---

## Task 2: `odin-recover` CLI wrapper

**Files:**
- Create: `tools/odin/asgard/recovery_cli.py`
- Test: `tools/odin/tests/test_recovery_cli.py`

Pattern matches `tools/odin/asgard/cli.py` (the `odin-dispatch` entry point): plain Python module with a `main()` that takes `argv`, prints `RecoveryResult.message`, exits 0 iff recovered. **No** console-script registration in `pyproject.toml` — invocation is `./isaaclab.sh -p tools/odin/asgard/recovery_cli.py`, matching `odin-dispatch`.

- [ ] **Step 2.1: Write failing tests** — `tools/odin/tests/test_recovery_cli.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.recovery_cli`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard import recovery_cli
from tools.odin.asgard.recovery import RecoveryResult


def _write_fleet_yaml(tmp_path: Path, host: str = "10.0.0.1") -> Path:
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text(
        f"""fleet_name: test-fleet
default_ssh_user: horde
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: {host}
    container_name: isaac-lab-base
"""
    )
    return fleet


def test_main_recovered_returns_zero(tmp_path, monkeypatch, capsys):
    fleet = _write_fleet_yaml(tmp_path, host="10.0.0.1")

    def _fake_recover(host, *, ssh):
        assert host.host == "10.0.0.1"
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={"docker_restart": "ok", "container_up": "ok", "gpu_probe": "ok"},
        )

    monkeypatch.setattr(recovery_cli, "recover_valkyrie_gpu", _fake_recover)
    rc = recovery_cli.main(["--fleet", str(fleet), "--host", "10.0.0.1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recovered_via_container_restart" in out


def test_main_not_recovered_returns_one(tmp_path, monkeypatch, capsys):
    fleet = _write_fleet_yaml(tmp_path, host="10.0.0.2")

    def _fake_recover(host, *, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=2.0,
            message="docker_restart_failed: daemon down",
            details={},
        )

    monkeypatch.setattr(recovery_cli, "recover_valkyrie_gpu", _fake_recover)
    rc = recovery_cli.main(["--fleet", str(fleet), "--host", "10.0.0.2"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "docker_restart_failed" in out


def test_main_unknown_host_errors(tmp_path, capsys):
    fleet = _write_fleet_yaml(tmp_path, host="10.0.0.1")
    with pytest.raises(SystemExit) as ei:
        recovery_cli.main(["--fleet", str(fleet), "--host", "10.0.0.99"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "10.0.0.99" in err
```

- [ ] **Step 2.2: Run tests, verify they FAIL**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_recovery_cli.py -v`
Expected: All three FAIL with `ModuleNotFoundError`.

- [ ] **Step 2.3: Implement `recovery_cli.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-recover — ad-hoc GPU-loss recovery wrapper around :func:`recover_valkyrie_gpu`.

Invoke from the repo root with ``PYTHONPATH=.``:

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/recovery_cli.py \\
        --fleet fleet.yaml \\
        --host 10.176.214.169

Exits 0 iff the host's container restarts and ``nvidia-smi -L`` lists at
least one GPU. Exits 1 otherwise. Exits 2 if the host is not in the
fleet.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.recovery import recover_valkyrie_gpu
from tools.odin.asgard.transport import ShellSSHRunner

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="odin-recover",
        description="Restart a Valkyrie's container and verify the GPU is visible.",
    )
    parser.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    parser.add_argument("--host", required=True, help="Host address as it appears in fleet.yaml.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    fleet = load_fleet(ns.fleet)
    matches = [h for h in fleet.hosts if h.host == ns.host]
    if not matches:
        print(f"odin-recover: host {ns.host!r} not found in {ns.fleet}", file=sys.stderr)
        sys.exit(2)
    host = matches[0]
    result = recover_valkyrie_gpu(host, ssh=ShellSSHRunner())
    print(f"odin-recover: host={host.host} container={host.container_name} ", end="")
    print(f"recovered={result.recovered} duration_s={result.duration_s:.1f} message={result.message}")
    return 0 if result.recovered else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.4: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_recovery_cli.py -v`
Expected: 3/3 PASS.

- [ ] **Step 2.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/recovery_cli.py tools/odin/tests/test_recovery_cli.py
git commit -m "Add odin-recover CLI for ad-hoc GPU-loss recovery

Thin wrapper over recover_valkyrie_gpu, mirrors the odin-dispatch
invocation pattern (./isaaclab.sh -p tools/odin/asgard/recovery_cli.py).
Exits 0 if the named host's container restarts and exposes a GPU; 1 if
recovery fails; 2 if the host is not in the fleet."
```

---

## Task 3: Hugin output-existence check

**Files:**
- Modify: `tools/odin/hugin/run.py:40-61` (`_run_phase`)
- Test: `tools/odin/tests/test_hugin.py`

The bug surfaced in T4.2 fleet validation: when Isaac Sim crashed silently with returncode 0 producing no output, Hugin reported `status="completed", exit_code=0`. The aggregator's strict whitelist later flagged these as failures, but the manifest already lied. Fix at the source.

- [ ] **Step 3.1: Write the failing test** — append to `tools/odin/tests/test_hugin.py`

```python
def test_hugin_silent_exit_zero_no_output_marks_failed(tmp_path, monkeypatch):
    """Subprocess exits 0 but writes no output JSON → phase status='failed'."""

    def _silent_exit_zero(cmd, *args, **kwargs):
        # Do NOT create the --schema_v1_output file.
        class R:
            returncode = 0
            stdout = b""
            stderr = b""

        return R()

    bundle_root = str(tmp_path)
    monkeypatch.setattr(hugin_run, "_subprocess_run", _silent_exit_zero)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hugin",
            "--task",
            "Isaac-Ant-Direct-v0",
            "--backend",
            "physx",
            "--seed",
            "42",
            "--num_envs",
            "64",
            "--max_iterations",
            "5",
            "--runs_root",
            bundle_root,
        ],
    )
    with pytest.raises(SystemExit) as ei:
        hugin_run.main()
    # Hugin exits non-zero because both phases were promoted to failed.
    assert ei.value.code != 0

    # Manifest should reflect the failure.
    bundles = [d for d in os.listdir(bundle_root) if d.startswith("rsl-rl_physx_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["status"] == "failed"
    assert m["phases"]["training"]["exit_code"] != 0
```

- [ ] **Step 3.2: Run test, verify it FAILS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py::test_hugin_silent_exit_zero_no_output_marks_failed -v`
Expected: FAIL — current code records `status="completed", exit_code=0` despite missing output.

- [ ] **Step 3.3: Modify `_run_phase`** in `tools/odin/hugin/run.py`

Replace lines 40–61 with:

```python
def _run_phase(cmd: list[str], bundle_dir: str, phase_name: str, output_json: str) -> ManifestPhase:
    """Run one subprocess phase; capture exit code, duration, and log tails on failure.

    Defines "completed" as: returncode == 0 AND ``output_json`` exists. A
    silent-exit-0 (subprocess exits 0 but writes no output) is a known
    failure mode for Isaac Sim crashes — promote it to ``status="failed"``
    with a derived non-zero exit code so the worker's classifier and the
    aggregator both pick it up.
    """
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    start = datetime.now(timezone.utc)
    completed = _subprocess_run(cmd, capture_output=True)
    end = datetime.now(timezone.utc)
    duration_s = (end - start).total_seconds()
    output_exists = os.path.exists(output_json)
    if completed.returncode != 0 or not output_exists:
        status = "failed"
        # Promote silent-exit-0 to a non-zero exit code so main() exits non-zero.
        exit_code = completed.returncode or 1
        with open(os.path.join(logs_dir, f"{phase_name}.stderr.log"), "wb") as f:
            f.write(tail_bytes(completed.stderr))
        with open(os.path.join(logs_dir, f"{phase_name}.stdout.log"), "wb") as f:
            f.write(tail_bytes(completed.stdout))
    else:
        status = "completed"
        exit_code = completed.returncode
    return ManifestPhase(
        file=os.path.basename(output_json),
        status=status,
        duration_s=duration_s,
        exit_code=exit_code,
    )
```

- [ ] **Step 3.4: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py -v`
Expected: ALL pass — both new test and the existing `test_hugin_happy_path` / `test_hugin_failure_path_writes_logs`.

- [ ] **Step 3.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/hugin/run.py tools/odin/tests/test_hugin.py
git commit -m "Hugin: treat silent-exit-zero as failed phase

_run_phase now requires returncode == 0 AND the expected output JSON
to exist before declaring 'completed'. Silent crashes (subprocess
exits 0 but writes no output — observed when Isaac Sim loses its GPU
mid-run) are promoted to status='failed' with derived exit_code=1, so
Hugin's main() propagates the failure and the worker classifier
records hugin_crash instead of a ghost completion."
```

---

## Task 4: Munin output-existence mirror

**Files:**
- Modify: `tools/odin/munin/run.py` (`_run_phase`)
- Test: `tools/odin/tests/test_munin.py`

Identical fix in Munin (the SKRL counterpart). Munin has its own `_run_phase` copy.

- [ ] **Step 4.1: Read Munin's current `_run_phase`** to confirm the same shape as Hugin's pre-T3 form.

Run: `grep -n "_run_phase" tools/odin/munin/run.py`
Expected: matches Hugin's old shape (returncode-only check).

- [ ] **Step 4.2: Write the failing test** — `tools/odin/tests/test_munin.py`

```python
def test_munin_silent_exit_zero_no_output_marks_failed(tmp_path, monkeypatch):
    """Subprocess exits 0 but writes no output JSON → phase status='failed'."""
    from tools.odin.munin import run as munin_run
    import json
    import os

    def _silent_exit_zero(cmd, *args, **kwargs):
        class R:
            returncode = 0
            stdout = b""
            stderr = b""

        return R()

    bundle_root = str(tmp_path)
    monkeypatch.setattr(munin_run, "_subprocess_run", _silent_exit_zero)
    monkeypatch.setattr(
        "sys.argv",
        [
            "munin",
            "--task",
            "Isaac-Ant-Direct-v0",
            "--backend",
            "physx",
            "--seed",
            "42",
            "--num_envs",
            "64",
            "--max_iterations",
            "5",
            "--runs_root",
            bundle_root,
        ],
    )
    import pytest

    with pytest.raises(SystemExit) as ei:
        munin_run.main()
    assert ei.value.code != 0

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("skrl_physx_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["status"] == "failed"
    assert m["phases"]["training"]["exit_code"] != 0
```

- [ ] **Step 4.3: Run test, verify it FAILS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_munin.py::test_munin_silent_exit_zero_no_output_marks_failed -v`
Expected: FAIL.

- [ ] **Step 4.4: Apply the same `_run_phase` fix** in `tools/odin/munin/run.py`

Use the same code block as Step 3.3 (the function body is identical between Hugin and Munin).

- [ ] **Step 4.5: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_munin.py -v`
Expected: ALL pass.

- [ ] **Step 4.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/munin/run.py tools/odin/tests/test_munin.py
git commit -m "Munin: treat silent-exit-zero as failed phase

Mirror of the Hugin fix: _run_phase requires returncode == 0 AND the
output JSON to exist before declaring 'completed'."
```

---

## Task 5: Preflight `gpu_present` check

**Files:**
- Modify: `tools/odin/asgard/preflight.py` (add 5th check)
- Test: `tools/odin/tests/test_asgard_preflight.py`

- [ ] **Step 5.1: Write failing tests** — append to `tools/odin/tests/test_asgard_preflight.py`

```python
def test_all_checks_pass_with_gpu():
    """Five-check happy path: ssh + docker + container + isaaclab + gpu."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=0,
                stdout="GPU 0: NVIDIA A100 (UUID: GPU-abc...)\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is True
    assert r.checks == {
        "ssh_reach": True,
        "docker_running": True,
        "container_up": True,
        "isaaclab_present": True,
        "gpu_present": True,
    }


def test_gpu_absent_marks_host_down():
    """nvidia-smi -L returns non-zero → preflight fails with gpu_present=False."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=255,
                stdout="",
                stderr="Failed to initialize NVML: Unknown Error",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["gpu_present"] is False
    assert "gpu" in r.message.lower()


def test_gpu_present_short_circuits_on_container_down():
    """If container_up fails, gpu_present stays False without probe call."""
    call_count = {"n": 0}

    class _CountingSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            call_count["n"] += 1
            if "echo preflight-ok" in cmd:
                return _ok()
            if "docker ps" in cmd:
                return _ok()
            if "docker inspect" in cmd:
                return SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.01)
            if "nvidia-smi -L" in cmd:
                # Should never reach here.
                raise AssertionError("nvidia-smi -L called despite container down")
            return _ok()

    r = preflight_valkyrie(_host(), ssh=_CountingSSH())
    assert r.ok is False
    assert r.checks["gpu_present"] is False
    assert r.checks["container_up"] is False
```

Update the existing `test_all_checks_pass` to include `gpu_present` in its scripted responses and assertions (otherwise it will start failing once the 5th check is added):

```python
def test_all_checks_pass():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=0,
                stdout="GPU 0: NVIDIA A100\n",
                stderr="",
                duration_s=0.01,
            ),
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
        "gpu_present": True,
    }
```

- [ ] **Step 5.2: Run tests, verify failures**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_preflight.py -v`
Expected: New three FAIL; existing `test_all_checks_pass` (after edit) FAILs because `checks` dict missing `gpu_present` key.

- [ ] **Step 5.3: Modify `preflight.py`** — add the 5th check

In `tools/odin/asgard/preflight.py`, update the `checks` dict initial value and add a Phase 5 block before the final return. The result diff:

```python
    checks = {
        "ssh_reach": False,
        "docker_running": False,
        "container_up": False,
        "isaaclab_present": False,
        "gpu_present": False,
    }
    # ... existing checks 1-4 unchanged ...

    # 5. gpu_present — at least one GPU visible inside the running container.
    r = ssh.run(host, f"docker exec {host.container_name} nvidia-smi -L", timeout_s=15.0)
    if r.exit_code != 0 or not r.stdout.strip():
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"GPU absent in container: {r.stderr.strip() or 'empty stdout'}",
        )
    checks["gpu_present"] = True

    return PreflightResult(host=host.host, ok=True, checks=checks, message="")
```

- [ ] **Step 5.4: Run tests, verify all PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_preflight.py -v`
Expected: ALL PASS (existing + 3 new).

- [ ] **Step 5.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/preflight.py tools/odin/tests/test_asgard_preflight.py
git commit -m "Preflight: add gpu_present check

Fifth check after isaaclab_present: docker exec <ctr> nvidia-smi -L
must exit 0 with at least one GPU listed. Catches hosts that arrive
at dispatch start with a wedged container GPU. Short-circuits on
prior-check failures (existing pattern)."
```

---

## Task 6: Document `gpu_lost` in `FailureInfo`

**Files:**
- Modify: `tools/odin/asgard/jobs.py` (docstring only)

The dataclass takes `kind: str`; the doc block enumerates known kinds. Add `gpu_lost`. No code change. No new tests (string-only documentation).

- [ ] **Step 6.1: Edit `tools/odin/asgard/jobs.py`** — extend the `FailureInfo` docstring

Replace the docstring of the `FailureInfo` dataclass with:

```python
@dataclass
class FailureInfo:
    """Classified failure attached to a :class:`JobEntry` when ``status == 'failed'``.

    ``kind`` values:

    - ``infrastructure``: docker / SSH transport failure (retried).
    - ``hugin_crash``: training process exited non-zero with no
      Odin-recognised stderr signal. Also covers Hugin's silent-exit-0
      case (returncode 0 but no output JSON), which ``_run_phase``
      promotes to a non-zero exit before ``main()`` returns.
    - ``hugin_malformed_bundle``: SSH succeeded, rsync pulled, but the
      bundle's manifest is missing or invalid.
    - ``timeout``: SSH wall-clock timeout fired.
    - ``preset_unsupported``: training process exited non-zero with a
      stderr line beginning ``preset_unsupported:`` — the requested
      preset doesn't exist for the task. Caught by the runtime safety
      net when yaml-stamped ``presets_available`` is stale.
    - ``gpu_lost``: training process exited non-zero with a GPU-loss
      signature in stderr (NVML init failure, CUDA "no device", Vulkan
      driver mismatch). Worker attempts container-restart-based
      recovery before retrying on the same host. Counts against
      ``max_infrastructure_retries``.
    """
```

- [ ] **Step 6.2: Run all Odin tests as a smoke check**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests -q`
Expected: Still green (docstring change is a no-op).

- [ ] **Step 6.3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/jobs.py
git commit -m "Document gpu_lost in FailureInfo kind enum

Doc-only change preceding the worker classifier. Adds 'gpu_lost' to
the FailureInfo kind enumeration."
```

---

## Task 7: Worker `_classify` — GPU-loss stderr matcher

**Files:**
- Modify: `tools/odin/asgard/worker.py` (`_classify`, module-level constant)
- Test: `tools/odin/tests/test_asgard_worker.py`

Pure detection layer. No retry-loop or recovery integration yet — `_classify` returns `FailureInfo(kind="gpu_lost")`; the existing retry loop sees a non-`infrastructure` failure and emits terminal failed (current behavior). T8 adds the recovery branch.

- [ ] **Step 7.1: Read existing `test_asgard_worker.py`** to find the test fixtures used for `_classify`.

Run: `grep -n "_classify\|FakeSSH\|test_classify" tools/odin/tests/test_asgard_worker.py | head -20`
Use the same fixture pattern these tests use (likely a fake SSH + a constructed `JobEntry`).

- [ ] **Step 7.2: Write failing tests** — append to `tools/odin/tests/test_asgard_worker.py`

```python
def test_classify_gpu_lost_signature_nvml(tmp_path):
    """Stderr containing 'Failed to initialize NVML' → kind='gpu_lost'."""
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_X_seed42",
        task_id="X",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_X_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_X_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=1,
        stdout="",
        stderr="Failed to initialize NVML: Unknown Error\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "gpu_lost"
    assert "GPU-loss signature" in failure.message


def test_classify_gpu_lost_signature_cuda(tmp_path):
    """Stderr containing 'CUDA error: no CUDA-capable device' → kind='gpu_lost'."""
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_Y_seed42",
        task_id="Y",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Y_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_Y_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=1,
        stdout="",
        stderr="RuntimeError: CUDA error: no CUDA-capable device is detected\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "gpu_lost"


def test_classify_gpu_lost_signature_vulkan(tmp_path):
    """Stderr containing 'Vulkan ERROR_INCOMPATIBLE_DRIVER' → kind='gpu_lost'."""
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_Z_seed42",
        task_id="Z",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Z_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_Z_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=1,
        stdout="",
        stderr="[error] Vulkan ERROR_INCOMPATIBLE_DRIVER: cannot create instance\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "gpu_lost"


def test_classify_no_false_positive_on_success(tmp_path):
    """Exit 0 + signature in stderr (warning) → _classify returns None."""
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_OK_seed42",
        task_id="OK",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_OK_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_OK_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=0,
        stdout="...",
        stderr="warning: Failed to initialize NVML (recoverable)\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is None


def test_classify_timeout_wins_over_gpu_signature(tmp_path):
    """timed_out=True + stderr has CUDA error → kind='timeout'."""
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=14400),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_TO_seed42",
        task_id="TO",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_TO_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_TO_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=-15,
        stdout="",
        stderr="CUDA error: no CUDA-capable device is detected\n",
        duration_s=14400.0,
    )
    r.timed_out = True
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "timeout"
```

- [ ] **Step 7.3: Run tests, verify they FAIL**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -k "classify_gpu_lost or no_false_positive or timeout_wins" -v`
Expected: 4/5 FAIL (the 4 gpu_lost ones — `timeout_wins` may already pass since timeout check runs first in current `_classify`).

- [ ] **Step 7.4: Modify `tools/odin/asgard/worker.py`**

Add the module-level constant after `_INFRASTRUCTURE_DOCKER_EXIT_CODES`:

```python
# GPU-loss stderr signatures recognised by ``_classify``.  Worker emits
# FailureInfo(kind="gpu_lost") when the training process exited non-zero
# AND its stderr contains any of these strings.  Recovery (T8) is then
# attempted via container restart before retrying on the same host.
_GPU_LOST_SIGNATURES = (
    "Failed to initialize NVML",
    "CUDA error: no CUDA-capable device is detected",
    "Vulkan ERROR_INCOMPATIBLE_DRIVER",
)
```

Modify `_classify` so the GPU-loss branch runs **before** `preset_unsupported`. Replace the `if r.exit_code != 0:` block in `_classify` with:

```python
        if r.exit_code != 0:
            stderr_text = r.stderr or ""
            if any(sig in stderr_text for sig in _GPU_LOST_SIGNATURES):
                return FailureInfo(
                    kind="gpu_lost",
                    message="GPU-loss signature in stderr",
                    details={
                        "exit_code": r.exit_code,
                        "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                    },
                )
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
```

- [ ] **Step 7.5: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v`
Expected: ALL PASS (existing + 5 new).

- [ ] **Step 7.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker.py
git commit -m "Worker: classify GPU-loss stderr signatures

_classify gains a branch matching three stderr signatures (NVML init
failure, CUDA 'no device', Vulkan ERROR_INCOMPATIBLE_DRIVER) and
emits FailureInfo(kind='gpu_lost'). Order: timeout > docker-infra >
gpu_lost > preset_unsupported > hugin_crash. Recovery integration
follows in the next commit."
```

---

## Task 8: Worker `_execute` — recovery integration

**Files:**
- Modify: `tools/odin/asgard/worker.py` (`_execute`, `StateEvent`)
- Test: `tools/odin/tests/test_asgard_worker.py`

Wires `gpu_lost` failures into a recovery branch: call `recover_valkyrie_gpu`, emit `recovered`/`host_down` event, decide whether to retry on same host or fall through to terminal failure.

- [ ] **Step 8.1: Write failing tests** — append to `tools/odin/tests/test_asgard_worker.py`

```python
def test_worker_gpu_lost_recovery_succeeds_retries_same_host(tmp_path, monkeypatch):
    """First attempt: gpu_lost stderr → recover succeeds → second attempt succeeds."""
    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    # Scripted SSH: first call (Hugin) fails with NVML; second call (Hugin
    # retry) succeeds with exit 0 + valid bundle.
    ssh_responses = [
        SSHResult(
            exit_code=1,
            stdout="",
            stderr="Failed to initialize NVML: Unknown Error\n",
            duration_s=12.0,
        ),
        SSHResult(exit_code=0, stdout="ok", stderr="", duration_s=600.0),
    ]

    class _SeqSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            return ssh_responses.pop(0)

    # Build a minimal valid bundle so _validate_bundle passes after retry.
    job = JobEntry(
        run_id="rsl-rl_physx_R_seed42",
        task_id="R",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_R_seed42",
    )
    bundle = tmp_path / job.bundle_dir_name
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"schema_version": "1.0"}')

    class _FakeRsync:
        def pull(self, host, remote, local):
            return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.1)

    recover_calls = {"n": 0}

    def _fake_recover(host, *, ssh):
        recover_calls["n"] += 1
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=12.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    state_chan: queue.Queue = queue.Queue()
    worker = worker_mod.ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=state_chan,
        dispatch_dir=tmp_path,
        options=worker_mod.WorkerOptions(),
        ssh=_SeqSSH(),
        rsync=_FakeRsync(),
        shutdown_event=threading.Event(),
    )
    worker._execute(job)

    assert job.status == "completed"
    assert job.attempts == 2
    assert recover_calls["n"] == 1
    transitions = []
    while not state_chan.empty():
        transitions.append(state_chan.get_nowait().transition)
    assert "recovered" in transitions
    assert "completed" in transitions


def test_worker_gpu_lost_recovery_fails_marks_host_down(tmp_path, monkeypatch):
    """First attempt: gpu_lost → recover returns recovered=False → host_down + terminal failure."""
    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    class _SingleSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            return SSHResult(
                exit_code=1,
                stdout="",
                stderr="CUDA error: no CUDA-capable device is detected\n",
                duration_s=10.0,
            )

    def _fake_recover(host, *, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=2.0,
            message="docker_restart_failed: daemon down",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    state_chan: queue.Queue = queue.Queue()
    job = JobEntry(
        run_id="rsl-rl_physx_F_seed42",
        task_id="F",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_F_seed42",
    )
    (tmp_path / job.bundle_dir_name / "logs").mkdir(parents=True)
    (tmp_path / job.bundle_dir_name / "logs" / "ssh-tail.log").write_text("")

    worker = worker_mod.ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=state_chan,
        dispatch_dir=tmp_path,
        options=worker_mod.WorkerOptions(),
        ssh=_SingleSSH(),
        rsync=None,
        shutdown_event=threading.Event(),
    )
    worker._execute(job)

    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "gpu_lost"
    assert "v1" in job.preferred_not
    transitions = []
    while not state_chan.empty():
        transitions.append(state_chan.get_nowait().transition)
    assert "host_down" in transitions
    assert "failed" in transitions


def test_worker_gpu_lost_three_in_a_row_terminal_failure(tmp_path, monkeypatch):
    """Three consecutive gpu_lost + recovery=True → terminal fail at attempt 3."""
    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.fleet import ValkyrieConfig
    import queue
    import threading

    nvml_fail = SSHResult(
        exit_code=1,
        stdout="",
        stderr="Failed to initialize NVML: Unknown Error\n",
        duration_s=10.0,
    )
    ssh_responses = [nvml_fail, nvml_fail, nvml_fail]

    class _SeqSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            return ssh_responses.pop(0)

    def _fake_recover(host, *, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    state_chan: queue.Queue = queue.Queue()
    job = JobEntry(
        run_id="rsl-rl_physx_T_seed42",
        task_id="T",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_T_seed42",
    )
    (tmp_path / job.bundle_dir_name / "logs").mkdir(parents=True)
    (tmp_path / job.bundle_dir_name / "logs" / "ssh-tail.log").write_text("")

    worker = worker_mod.ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=state_chan,
        dispatch_dir=tmp_path,
        # max_infrastructure_retries=2 → up to 3 attempts.
        options=worker_mod.WorkerOptions(max_infrastructure_retries=2),
        ssh=_SeqSSH(),
        rsync=None,
        shutdown_event=threading.Event(),
    )
    worker._execute(job)

    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "gpu_lost"
    assert job.attempts == 3
```

- [ ] **Step 8.2: Run tests, verify they FAIL**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -k "gpu_lost_recovery or three_in_a_row" -v`
Expected: 3/3 FAIL — the recovery branch doesn't exist yet.

- [ ] **Step 8.3: Modify `tools/odin/asgard/worker.py`** — extend `_execute` retry loop and `StateEvent`

Add to imports:

```python
from tools.odin.asgard.recovery import recover_valkyrie_gpu
```

Update `StateEvent.transition` docstring (no struct change):

```python
@dataclass
class StateEvent:
    """Message posted by a worker to the state channel on every transition.

    ``transition`` values:

    - ``running``: job dispatched to host; ``started_at`` set.
    - ``completed``: job finished, bundle pulled, manifest validated.
    - ``failed``: terminal failure for this job; ``failure`` set.
    - ``recovered``: GPU loss detected, recovery succeeded; retry follows.
    - ``host_down``: GPU loss detected, recovery failed; host transitions
      to ``status="down"``; runner removes it from the worker pool.
    - ``shutdown_idle``: worker received its sentinel and exited cleanly.
    """

    run_id: str
    host: str
    transition: str
    failure: FailureInfo | None = None
    started_at: str | None = None
    ended_at: str | None = None
```

Replace the existing retry-loop tail in `_execute` (the `if failure.kind == "infrastructure"` block) with the extended version:

```python
            failure = self._classify(ssh_result, job, ssh_tail)
            if failure is not None and failure.kind == "infrastructure":
                if job.attempts <= self._options.max_infrastructure_retries:
                    continue
                # Exhausted retries → emit terminal failure below.
            elif failure is not None and failure.kind == "gpu_lost":
                rec = recover_valkyrie_gpu(self.host, ssh=self._ssh)
                if rec.recovered:
                    self._state_chan.put(
                        StateEvent(
                            run_id=job.run_id,
                            host=self.host.host,
                            transition="recovered",
                        )
                    )
                    if job.attempts <= self._options.max_infrastructure_retries:
                        continue
                    # Retries exhausted on this host even after successful recovery.
                else:
                    self._state_chan.put(
                        StateEvent(
                            run_id=job.run_id,
                            host=self.host.host,
                            transition="host_down",
                            failure=failure,
                        )
                    )
                    job.preferred_not = set(job.preferred_not) | {self.host.host}
                # Fall through to terminal failure emission below.

            break  # non-recoverable result or retries exhausted
```

- [ ] **Step 8.4: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v`
Expected: ALL PASS (existing + 3 new from this task + 5 from Task 7).

- [ ] **Step 8.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker.py
git commit -m "Worker: auto-recover GPU loss via container restart

_execute now branches on FailureInfo(kind='gpu_lost'): calls
recover_valkyrie_gpu and either retries on the same host (counts
against max_infrastructure_retries) or marks the host down via a
host_down StateEvent + preferred_not update on this job.  Three
consecutive gpu_lost failures on the same host for the same job
escalate to terminal failure."
```

---

## Task 9: Runner — consume `recovered` / `host_down` events

**Files:**
- Modify: `tools/odin/asgard/runner.py` (the `state_chan` consumer)
- Test: `tools/odin/tests/test_asgard_runner.py`

Runner currently rewrites `dispatch.json` on every `StateEvent`. Extend the handler so:

- `recovered` → set `FleetSnapshot[host].last_error = "gpu_lost: recovered"`. Status unchanged.
- `host_down` → set `FleetSnapshot[host].status = "down"`, `last_error = f"gpu_lost: recovery_failed ({failure.message})"`. Worker pool removal happens via existing logic since worker exits its loop after emitting terminal failure for the in-flight job (no new wiring needed for pool teardown).

- [ ] **Step 9.1: Locate the `state_chan` consumer**

Run: `grep -n "state_chan.get\|StateEvent\|fleet\[" tools/odin/asgard/runner.py | head -30`
Expected: hits inside the runner's main loop where it dequeues events.

- [ ] **Step 9.2: Write failing test** — append to `tools/odin/tests/test_asgard_runner.py`

```python
def test_runner_handles_recovered_event(tmp_path):
    """Recovered event updates fleet[host].last_error but not status."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.state import DispatchState, FleetSnapshot, SCHEMA_VERSION
    from tools.odin.asgard.worker import StateEvent

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[],
    )
    ev = StateEvent(run_id="r1", host="v1", transition="recovered")
    runner_mod._apply_state_event(state, ev)
    fs = next(f for f in state.fleet if f.host == "v1")
    assert fs.status == "busy"  # unchanged
    assert fs.last_error == "gpu_lost: recovered"


def test_runner_handles_host_down_event(tmp_path):
    """host_down event marks host status='down' with structured last_error."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.state import DispatchState, FleetSnapshot, SCHEMA_VERSION
    from tools.odin.asgard.worker import StateEvent
    from tools.odin.asgard.jobs import FailureInfo

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[],
    )
    ev = StateEvent(
        run_id="r1",
        host="v1",
        transition="host_down",
        failure=FailureInfo(kind="gpu_lost", message="docker_restart_failed: daemon down"),
    )
    runner_mod._apply_state_event(state, ev)
    fs = next(f for f in state.fleet if f.host == "v1")
    assert fs.status == "down"
    assert "gpu_lost: recovery_failed" in fs.last_error
    assert "docker_restart_failed" in fs.last_error
```

- [ ] **Step 9.3: Run tests, verify they FAIL**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_runner.py -k "recovered or host_down" -v`
Expected: FAIL — likely `_apply_state_event` doesn't exist (or doesn't recognise the transitions).

- [ ] **Step 9.4: Modify `tools/odin/asgard/runner.py`**

Locate the existing event handler (the block that updates `FleetSnapshot` based on `transition`). If it's an inline `if/elif` chain inside the main loop, **extract** it into a module-level helper named `_apply_state_event(state, ev)` (the test imports this name). Add the two new branches:

```python
def _apply_state_event(state: DispatchState, ev: StateEvent) -> None:
    """Mutate ``state`` in place to reflect one StateEvent transition.

    Called by the runner's main thread after dequeueing each event from
    ``state_chan``.  Centralises every transition's effect on
    ``FleetSnapshot`` and ``JobEntry`` so the dispatch.json rewrite
    happens off a single, easily-tested function.
    """
    fs = next((f for f in state.fleet if f.host == ev.host), None)
    if ev.transition == "running":
        if fs is not None:
            fs.status = "busy"
            fs.current_run_id = ev.run_id
    elif ev.transition == "completed":
        if fs is not None:
            fs.status = "idle"
            fs.current_run_id = None
        for j in state.jobs:
            if j.run_id == ev.run_id:
                j.status = "completed"
                j.ended_at = ev.ended_at
                break
    elif ev.transition == "failed":
        if fs is not None and fs.current_run_id == ev.run_id:
            fs.status = "idle"
            fs.current_run_id = None
        for j in state.jobs:
            if j.run_id == ev.run_id:
                j.status = "failed"
                j.failure = ev.failure
                j.ended_at = ev.ended_at
                break
    elif ev.transition == "recovered":
        if fs is not None:
            fs.last_error = "gpu_lost: recovered"
    elif ev.transition == "host_down":
        if fs is not None:
            fs.status = "down"
            detail = ev.failure.message if ev.failure is not None else "unknown"
            fs.last_error = f"gpu_lost: recovery_failed ({detail})"
            # current_run_id stays — worker is about to emit terminal "failed"
            # for the in-flight job in a follow-up event.
    elif ev.transition == "shutdown_idle":
        if fs is not None and fs.status != "down":
            fs.status = "idle"
            fs.current_run_id = None
```

> **Note:** The original handler may already have its own subtleties (e.g. how it updates `JobEntry` on `failed`). Keep the existing branches' behaviour byte-for-byte; only ADD the two new branches. If the handler is currently inlined inside the main runner loop, extract it without changing its semantics, then add the new branches.

- [ ] **Step 9.5: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_runner.py -v`
Expected: ALL PASS — including any pre-existing runner tests, since you preserved their behaviour during extraction.

- [ ] **Step 9.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py tools/odin/tests/test_asgard_runner.py
git commit -m "Runner: handle recovered / host_down state events

Extract the StateEvent → DispatchState mutation into a module-level
_apply_state_event helper.  Add two new transitions:
- recovered: sets fleet[host].last_error='gpu_lost: recovered'
- host_down: sets fleet[host].status='down' and last_error=
              f'gpu_lost: recovery_failed ({failure.message})'"
```

---

## Task 10: State schema 1.3

**Files:**
- Modify: `tools/odin/asgard/state.py:34` (constant only)
- Test: `tools/odin/tests/test_asgard_state.py`

Schema bump is purely versioning. The serializer already round-trips arbitrary `kind` strings and free-form `last_error`; no struct change.

- [ ] **Step 10.1: Write failing tests** — append to `tools/odin/tests/test_asgard_state.py`

```python
def test_schema_version_is_1_3():
    from tools.odin.asgard.state import SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.3"


def test_resume_from_1_2_state_works(tmp_path):
    """A 1.2 dispatch.json on disk is readable by the 1.3 reader (major-match)."""
    import json
    from tools.odin.asgard.state import read_dispatch_state

    payload = {
        "schema_version": "1.2",
        "dispatch_id": "20260424-160119",
        "started_at": "2026-04-24T16:01:19Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "abc123",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": [],
        "skipped": [],
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload, indent=2))
    state = read_dispatch_state(tmp_path)
    assert state.dispatch_id == "20260424-160119"
    # 1.2 file's schema_version field is preserved as-is on read.
    assert state.schema_version == "1.2"


def test_failure_kind_gpu_lost_round_trips(tmp_path):
    """JobEntry.failure with kind='gpu_lost' survives write→read."""
    j = _job("run-gl", status="failed")
    j.failure = FailureInfo(
        kind="gpu_lost",
        message="GPU-loss signature in stderr",
        details={
            "exit_code": 1,
            "log_tail_path": "run-gl/logs/ssh-tail.log",
        },
    )
    j.attempts = 2
    write_dispatch_state(tmp_path, _state([j]))

    reloaded = read_dispatch_state(tmp_path)
    rj = reloaded.jobs[0]
    assert rj.failure is not None
    assert rj.failure.kind == "gpu_lost"
    assert rj.failure.details["exit_code"] == 1
    assert rj.attempts == 2
```

> **Note:** `_state(...)` in the existing test file currently creates `DispatchState(schema_version="1.0", ...)`. Update its literal to `SCHEMA_VERSION` so the helper picks up future bumps automatically:
>
> ```python
> def _state(jobs: list[JobEntry]) -> DispatchState:
>     return DispatchState(
>         schema_version=SCHEMA_VERSION,
>         ...
>     )
> ```

- [ ] **Step 10.2: Run tests, verify they FAIL**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_state.py -v`
Expected: `test_schema_version_is_1_3` FAILS (current is 1.2). The other two new tests should already pass (no struct change).

- [ ] **Step 10.3: Bump `SCHEMA_VERSION`** in `tools/odin/asgard/state.py`

```python
SCHEMA_VERSION = "1.3"
```

- [ ] **Step 10.4: Run tests, verify they PASS**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_state.py -v`
Expected: ALL PASS, including any prior tests that referenced the version constant.

- [ ] **Step 10.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/state.py tools/odin/tests/test_asgard_state.py
git commit -m "Bump dispatch.json schema 1.2 -> 1.3

Additive only: failure.kind enum gains 'gpu_lost'; fleet[].last_error
may now contain 'gpu_lost: ...' prefixes.  Validators do major-match,
so 1.2 dispatches resume on 1.3 code unchanged."
```

---

## Task 11: End-to-end integration test + arch-doc update

**Files:**
- Modify: `tools/odin/tests/test_asgard_integration.py`
- Modify: `docs/odin/architecture.md`

Locks in the full detect-and-recover flow with a slow-marked test. Updates the architecture change-log with this iteration's entry.

- [ ] **Step 11.1: Write the slow-marked integration test** — append to `tools/odin/tests/test_asgard_integration.py`

The existing integration test uses two fixtures: `stub_ssh_runner` (monkeypatches `worker_mod._build_docker_exec_cmd` to a shell command that materialises a valid bundle) and `stub_provisioner` (monkeypatches preflight + provisioner to pass-through). Build a parallel fixture `stub_ssh_runner_first_job_nvml` that materialises a *failing* run for the first job per host (exits non-zero with NVML stderr) then a successful bundle on retry. Then monkeypatch `worker_mod.recover_valkyrie_gpu` to always return recovered=True. Add this test:

```python
@pytest.fixture
def stub_ssh_runner_first_job_nvml(monkeypatch, tmp_path):
    """Like stub_ssh_runner, but the FIRST docker-exec call per host fails
    with NVML stderr; subsequent calls succeed and materialise a bundle."""
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd
    seen_per_host: dict[str, int] = {}

    def _fake_build(host: ValkyrieConfig, job) -> str:
        seen = seen_per_host.get(host.host, 0)
        seen_per_host[host.host] = seen + 1
        if seen == 0:
            # First call on this host: fail with NVML stderr (exit 1).
            return "echo 'Failed to initialize NVML: Unknown Error' 1>&2 && exit 1"
        # Subsequent calls: materialise a valid bundle and exit 0.
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
    yield seen_per_host
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


def test_loopback_dispatch_recovers_from_gpu_lost(tmp_path, stub_ssh_runner_first_job_nvml, stub_provisioner, monkeypatch):
    """End-to-end: host's first job triggers gpu_lost stderr → fake recover
    returns recovered=True → second attempt completes the job. Final
    dispatch.json shows job completed, attempts=2, and
    fleet[host].last_error == 'gpu_lost: recovered'."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult

    recover_calls: list[str] = []

    def _fake_recover(host, *, ssh):
        recover_calls.append(host.host)
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    # Build a one-row env list (same shape as test_loopback_dispatch_against_localhost).
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=False,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        ),
    ]
    physx_yaml = tmp_path / "physx_envs.yaml"
    write_env_list(physx_yaml, el)

    fleet = Fleet(
        fleet_name="loopback",
        hosts=[ValkyrieConfig(host="localhost", ssh_user=os.environ.get("USER", "ci"), isaaclab_path=str(tmp_path))],
    )
    options = DispatchOptions(seeds=[42], physx_yaml=physx_yaml, runs_root=tmp_path / "runs")
    run_dispatch(fleet=fleet, options=options, ssh=ShellSSHRunner(), rsync=ShellRsyncRunner())

    # Read final dispatch.json.
    dispatch_dirs = sorted([p for p in (tmp_path / "runs").iterdir() if p.is_dir()])
    assert len(dispatch_dirs) == 1
    dispatch_json = json.loads((dispatch_dirs[0] / "dispatch.json").read_text())
    job = dispatch_json["jobs"][0]
    assert job["status"] == "completed"
    assert job["attempts"] == 2
    fleet_entry = next(f for f in dispatch_json["fleet"] if f["host"] == "localhost")
    assert fleet_entry["last_error"] == "gpu_lost: recovered"
    assert recover_calls == ["localhost"]
```

> **Note:** `DispatchOptions` may have additional required fields by the time this lands. Mirror the existing `test_loopback_dispatch_against_localhost` test's call signature exactly — copy the call from that test and append `runs_root=tmp_path / "runs"` if not already there. Don't introduce new constructor params.

- [ ] **Step 11.2: Run the test, verify it PASSES**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_integration.py -m slow -v`
Expected: New test PASSES. (May take ~2–10 s depending on monkeypatching of `time.sleep`.)

- [ ] **Step 11.3: Update `docs/odin/architecture.md`**

In the `## Change log` section, prepend a new dated entry:

```markdown
- **2026-04-27** — GPU-loss detection + recovery feature (3 layers,
  schema 1.2 → 1.3 additive).  Hugin/Munin `_run_phase` now require
  output JSON to exist before declaring `completed` (silent-exit-0 is
  promoted to `failed`).  Worker `_classify` recognises NVML / CUDA /
  Vulkan stderr signatures and emits `FailureInfo(kind="gpu_lost")`.
  New module `tools.odin.asgard.recovery` does container-restart-based
  GPU recovery; worker auto-runs it between `gpu_lost` retries.  A
  preflight `gpu_present` check catches hosts that arrive at dispatch
  start with a wedged container GPU.  New CLI `odin-recover` exposes
  the recovery tool for ad-hoc operator use.
```

Update the "Last updated" line at the top of the doc to `2026-04-27 (Odin GPU-loss recovery)`.

- [ ] **Step 11.4: Run the full Odin test suite once more as a smoke check**

Run: `./isaaclab.sh -p -m pytest tools/odin/tests -q`
Expected: ALL PASS.

- [ ] **Step 11.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_integration.py docs/odin/architecture.md
git commit -m "GPU-loss recovery: end-to-end integration test + arch doc

Slow-marked integration test exercises the full detect-and-recover
flow through run_dispatch with monkeypatched recovery.  Architecture
change-log entry summarising the four-layer feature."
```

---

## Self-review (run after writing — already done in the plan above)

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| § Architecture (4 layers) | T1, T3-4, T5, T7-8 (one per layer) |
| § Components → recovery.py | T1 |
| § Components → recovery_cli.py | T2 |
| § Components → Hugin _run_phase | T3 |
| § Components → Munin _run_phase | T4 |
| § Components → preflight gpu_present | T5 |
| § Components → FailureInfo doc | T6 |
| § Components → Worker `_classify` | T7 |
| § Components → Worker `_execute` recovery integration | T8 |
| § Components → state.py SCHEMA_VERSION 1.3 | T10 |
| § Components → runner.py StateEvent handler | T9 |
| § Data flow Flow 1 (healthy) | unchanged — no test needed |
| § Data flow Flow 2 (silent-exit) | T3 + T4 |
| § Data flow Flow 3a (recover succeeds) | T8 first integration test |
| § Data flow Flow 3b (recover fails) | T8 second integration test |
| § Data flow Flow 4 (preflight catches) | T5 |
| § Error handling matrix (5 cases) | T1 (5 tests) |
| § Error handling — three-in-a-row | T8 third integration test |
| § Error handling — schema migration | T10 |
| § Error handling — concurrency | by construction (one worker per host) |
| § Error handling — logging recovery.log | T8 wiring (recovery_cli main passes through; worker integration logs to ssh-tail.log + worker prints) |
| § Testing strategy (6 layers) | T1, T5, T7+T8, T3+T4, T10, T11 |

**Placeholder scan:** searched for "TBD", "TODO", "fill in", "<...>" — none. Every code step has concrete code or a precise edit.

**Type/signature consistency:**

- `recover_valkyrie_gpu(host, *, ssh) -> RecoveryResult` — used identically in T1 (definition), T2 (CLI), T8 (worker monkeypatch).
- `RecoveryResult` fields used in T2, T8 match T1 definition: `host, container_name, attempted, recovered, duration_s, message, details`.
- `FailureInfo(kind="gpu_lost", message=..., details=...)` shape consistent across T7, T8, T9, T10.
- `StateEvent(run_id, host, transition, failure?, started_at?, ended_at?)` — new transitions `recovered` and `host_down` declared in T8, consumed in T9.
- `_GPU_LOST_SIGNATURES` constant defined in T7, used only in T7's `_classify` body. Same string list as the spec's signature table.

Plan is internally consistent and spec-complete.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-04-27-odin-gpu-loss-recovery.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec compliance then code quality) between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
