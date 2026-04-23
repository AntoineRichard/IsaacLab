# Odin Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `odin-bootstrap --fleet fleet.yaml` — a CLI that brings a fresh Valkyrie from "just SSH + Docker + GPU" to T3.1-preflight-ready state (IsaacLab rsync'd + `isaac-lab-base` container running), so the existing T3.1 dispatch path works against truly fresh hardware.

**Architecture:** One new module `tools/odin/asgard/bootstrap.py` with `bootstrap_valkyrie` (per-host, 6-step pipeline) + `bootstrap_fleet` (parallel driver). One new CLI at `tools/odin/asgard/bootstrap_cli.py`. One tiny provisioner tweak (`_container_start` gains a `timeout_s` param) so the 30-min docker build fits. Zero changes to T3.1's dispatch / preflight / worker / state machine.

**Tech Stack:** Python 3.10+, stdlib only (`concurrent.futures`, `dataclasses`, `time`, `argparse`, `pathlib`), pytest.

**Spec:** `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md`

---

## File Structure

**New files:**
- `tools/odin/asgard/bootstrap.py` — `BootstrapResult`, `bootstrap_valkyrie`, `bootstrap_fleet`.
- `tools/odin/asgard/bootstrap_cli.py` — argparse + `main()` for `odin-bootstrap`.
- `tools/odin/tests/test_asgard_bootstrap.py` — unit tests for `bootstrap_valkyrie` + `bootstrap_fleet`.
- `tools/odin/tests/test_asgard_bootstrap_cli.py` — CLI-level tests.

**Modified:**
- `tools/odin/asgard/provisioner.py` — `_container_start` gains a `timeout_s` keyword parameter (default 300 s).
- `tools/odin/asgard/__init__.py` — re-export the 3 new public bootstrap symbols.
- `tools/odin/README.md` — new "Bootstrapping a fresh fleet" section.

**Unchanged (confirmed):**
- `tools/odin/asgard/runner.py` (T3.1 `run_dispatch`).
- `tools/odin/asgard/preflight.py`.
- `tools/odin/asgard/worker.py`.
- `tools/odin/asgard/state.py`.
- `tools/odin/asgard/transport.py` (ShellSSHRunner/ShellRsyncRunner signatures).

**Task ordering rationale:** Task 1 is the prerequisite tweak (1-param change) with its own tiny test. Task 2 builds the per-host `bootstrap_valkyrie` + its 8 failure-path tests. Task 3 adds `bootstrap_fleet` (parallelism). Task 4 ships the CLI. Task 5 wires `__init__.py` re-exports. Task 6 updates the README. Task 7 is the final-suite + architecture-doc sweep.

---

### Task 1: Provisioner `_container_start` timeout param

**Files:**
- Modify: `tools/odin/asgard/provisioner.py`
- Modify: `tools/odin/tests/test_asgard_provisioner.py`

- [ ] **Step 1: Read the current provisioner code to locate the function**

```bash
grep -n "_container_start\|def _container" tools/odin/asgard/provisioner.py
```

Expected: a `_container_start(host, ssh) -> bool` function around lines 59-65 that hardcodes `timeout_s=300.0` inside its `ssh.run(...)` call.

- [ ] **Step 2: Write the failing test**

Append to `tools/odin/tests/test_asgard_provisioner.py`:

```python
def test_container_start_respects_custom_timeout():
    """`_container_start(timeout_s=1800)` must reach the SSH call as timeout_s=1800."""
    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.provisioner import _container_start

    recorded: list[dict] = []

    class _RecordingSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            recorded.append({"cmd": cmd, "timeout_s": timeout_s})

            class R:
                exit_code = 0
                stdout = ""
                stderr = ""
                duration_s = 0.1

            return R()

    host = ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
    )
    ok = _container_start(host, _RecordingSSH(), timeout_s=1800)
    assert ok is True
    assert len(recorded) == 1
    assert recorded[0]["timeout_s"] == 1800
    assert "container.py start" in recorded[0]["cmd"]


def test_container_start_default_timeout_is_300():
    """Calling `_container_start(host, ssh)` without `timeout_s` keeps the warm-path default."""
    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.provisioner import _container_start

    recorded: list[dict] = []

    class _RecordingSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            recorded.append({"timeout_s": timeout_s})

            class R:
                exit_code = 0
                stdout = ""
                stderr = ""
                duration_s = 0.1

            return R()

    host = ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
    )
    _container_start(host, _RecordingSSH())
    assert recorded[0]["timeout_s"] == 300
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run:
```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_provisioner.py::test_container_start_respects_custom_timeout tools/odin/tests/test_asgard_provisioner.py::test_container_start_default_timeout_is_300 -v --confcutdir=tools/odin
```

Expected: FAIL on `test_container_start_respects_custom_timeout` (the keyword argument doesn't yet exist — `TypeError: _container_start() got an unexpected keyword argument 'timeout_s'`). The default-timeout test may pass incidentally depending on the current hardcoded value.

- [ ] **Step 4: Edit `_container_start` to accept a `timeout_s` parameter**

In `tools/odin/asgard/provisioner.py`, find the current `_container_start`:

```python
def _container_start(host: ValkyrieConfig, ssh: SSHRunner) -> bool:
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py start",
        timeout_s=300.0,
    )
    return r.exit_code == 0
```

Replace with:

```python
def _container_start(host: ValkyrieConfig, ssh: SSHRunner, *, timeout_s: int = 300) -> bool:
    """Run ``./docker/container.py start`` on ``host`` and return True on success.

    The warm-path default of 300 s suits subsequent dispatches where the
    container image is already built. First-time bootstrap must pass a
    longer ``timeout_s`` (see :mod:`tools.odin.asgard.bootstrap`) because the
    first-time docker build takes 15-30 minutes.
    """
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py start",
        timeout_s=timeout_s,
    )
    return r.exit_code == 0
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_provisioner.py -v --confcutdir=tools/odin
```

Expected: all provisioner tests pass (existing + 2 new).

- [ ] **Step 6: Run pre-commit + commit**

```bash
./isaaclab.sh -f
# If hooks modified files, re-stage and re-run until clean.
git add tools/odin/asgard/provisioner.py tools/odin/tests/test_asgard_provisioner.py
git commit -m "Provisioner: make _container_start timeout configurable"
```

Subject is 50 chars — exactly at the cap.

---

### Task 2: `bootstrap_valkyrie` — per-host bootstrap pipeline

**Files:**
- Create: `tools/odin/asgard/bootstrap.py`
- Create: `tools/odin/tests/test_asgard_bootstrap.py`

- [ ] **Step 1: Write the failing tests (single-host path only)**

Create `tools/odin/tests/test_asgard_bootstrap.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla-adjacent Asgard bootstrap (bring fresh Valkyries up)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.asgard.bootstrap import BootstrapResult, bootstrap_valkyrie
from tools.odin.asgard.fleet import ValkyrieConfig


# --- Fakes -----------------------------------------------------------------


@dataclass
class _SSHCall:
    cmd: str
    timeout_s: float | None


@dataclass
class _RsyncCall:
    local_path: Path
    remote_path: str


@dataclass
class _FakeSSH:
    """Records calls; replies with a per-call exit_code lookup.

    Default reply is exit_code=0. Override by setting ``replies[key]`` where
    ``key`` is a substring that must appear in the cmd. First match wins;
    check order follows insertion order.
    """

    calls: list[_SSHCall] = field(default_factory=list)
    replies: dict[str, int] = field(default_factory=dict)
    reply_stdout: dict[str, str] = field(default_factory=dict)
    reply_stderr: dict[str, str] = field(default_factory=dict)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
        self.calls.append(_SSHCall(cmd=cmd, timeout_s=timeout_s))
        exit_code = 0
        stdout = ""
        stderr = ""
        for key, code in self.replies.items():
            if key in cmd:
                exit_code = code
                stdout = self.reply_stdout.get(key, "")
                stderr = self.reply_stderr.get(key, "")
                break

        class R:
            pass

        R.exit_code = exit_code
        R.stdout = stdout
        R.stderr = stderr
        R.duration_s = 0.01
        return R()


@dataclass
class _FakeRsync:
    calls: list[_RsyncCall] = field(default_factory=list)
    exit_code: int = 0
    stderr: str = ""

    def push(self, host, local_path, remote_path):
        self.calls.append(_RsyncCall(local_path=Path(local_path), remote_path=str(remote_path)))

        class R:
            pass

        R.exit_code = self.exit_code
        R.stdout = ""
        R.stderr = self.stderr
        R.duration_s = 0.01
        return R()

    def pull(self, host, remote_path, local_path):
        raise AssertionError("bootstrap_valkyrie must not call rsync.pull")


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
        container_name="isaac-lab-base",
    )


# --- Tests -----------------------------------------------------------------


def test_bootstrap_valkyrie_happy_path(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert isinstance(result, BootstrapResult)
    assert result.ok is True
    assert result.host == "v1.internal"
    assert set(result.step_durations_s.keys()) == {"wipe", "rsync", "container_start", "container_verify"}
    assert all(d >= 0.0 for d in result.step_durations_s.values())


def test_bootstrap_valkyrie_ssh_unreachable(tmp_path: Path):
    ssh = _FakeSSH(replies={"echo bootstrap-ok": 255})
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "ssh unreachable" in result.message
    assert rsync.calls == [], "rsync.push must not run when ssh is down"


def test_bootstrap_valkyrie_docker_daemon_down(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker ps": 1},
        reply_stderr={"docker ps": "Cannot connect to the Docker daemon"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "docker daemon" in result.message
    assert rsync.calls == []


def test_bootstrap_valkyrie_rsync_failure(tmp_path: Path):
    ssh = _FakeSSH()  # all ssh ok
    rsync = _FakeRsync(exit_code=23, stderr="send_files failed")
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "rsync push failed" in result.message
    # Wipe ran; rsync ran; container_start did NOT.
    assert any("rm -rf" in c.cmd for c in ssh.calls)
    assert len(rsync.calls) == 1
    assert not any("container.py start" in c.cmd for c in ssh.calls)


def test_bootstrap_valkyrie_container_start_failure(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"container.py start": 1},
        reply_stderr={"container.py start": "timeout"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "container.py start" in result.message or "container start failed" in result.message
    assert not any("docker inspect" in c.cmd for c in ssh.calls), "verify must not run when start failed"


def test_bootstrap_valkyrie_container_not_running_after_start(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "exited"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "not running" in result.message
    assert "'exited'" in result.message


def test_bootstrap_valkyrie_build_timeout_passed_through(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync, build_timeout_s=1800)
    start_calls = [c for c in ssh.calls if "container.py start" in c.cmd]
    assert len(start_calls) == 1
    assert start_calls[0].timeout_s == 1800


def test_bootstrap_valkyrie_wipe_always_runs(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    wipe_calls = [c for c in ssh.calls if "rm -rf" in c.cmd and "/opt/IsaacLab" in c.cmd]
    assert len(wipe_calls) == 1


def test_bootstrap_valkyrie_wipe_failure(tmp_path: Path):
    """Wipe-step failure (e.g. permission issue) halts the pipeline before rsync."""
    ssh = _FakeSSH(
        replies={"rm -rf": 1},
        reply_stderr={"rm -rf": "Permission denied"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "failed to wipe" in result.message
    assert "/opt/IsaacLab" in result.message
    assert rsync.calls == [], "rsync.push must not run when wipe failed"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap.py -v --confcutdir=tools/odin
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tools.odin.asgard.bootstrap'`.

- [ ] **Step 3: Implement `bootstrap_valkyrie` + supporting code**

Create `tools/odin/asgard/bootstrap.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh-Valkyrie bootstrap — bring a naked host to T3.1-preflight-ready state.

Unlike :mod:`tools.odin.asgard.provisioner` (the warm-path refresher used
inside :func:`~tools.odin.asgard.runner.run_dispatch`), bootstrap assumes the
remote has *only* SSH + Docker + a GPU — no IsaacLab clone, no container.
It wipes any prior tree, pushes the working tree, boots the Isaac Lab
container with a long enough timeout to survive first-time image build,
and verifies the container ended up in ``"running"`` state.
"""

from __future__ import annotations

import concurrent.futures as _cf
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.provisioner import _container_start
from tools.odin.asgard.transport import RsyncRunner, SSHRunner

__all__ = ["BootstrapResult", "bootstrap_valkyrie", "bootstrap_fleet"]


@dataclass
class BootstrapResult:
    """Outcome of bootstrapping a single Valkyrie."""

    host: str
    ok: bool
    message: str = ""
    commit_sha: str = ""
    step_durations_s: dict[str, float] = field(default_factory=dict)


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


def _time_step() -> float:
    return time.perf_counter()


def bootstrap_valkyrie(
    host: ValkyrieConfig,
    working_tree: Path,
    *,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    build_timeout_s: int = 1800,
) -> BootstrapResult:
    """Bring a fresh Valkyrie to T3.1-preflight-ready state.

    Pipeline (short-circuits on any step failure):

      1. SSH reach — a 15 s ``echo bootstrap-ok`` probe.
      2. Docker daemon reach — a 15 s ``docker ps`` probe.
      3. Wipe — ``rm -rf {isaaclab_path}`` (always, for idempotent re-runs).
      4. Rsync — push ``working_tree`` to ``{isaaclab_path}``.
      5. Container start — ``./docker/container.py start`` with
         ``build_timeout_s``.
      6. Container verify — ``docker inspect`` must report ``"running"``.

    Args:
        host: Target Valkyrie.
        working_tree: Controller-side IsaacLab path to push.
        ssh: SSH runner.
        rsync: Rsync runner.
        build_timeout_s: Timeout [s] for ``./docker/container.py start``
            (default 1800 = 30 min; covers a cold first-time docker build).

    Returns:
        :class:`BootstrapResult` with ``ok=True`` iff all six steps passed.
        ``step_durations_s`` records wall-clock seconds for steps 3-6 (the
        ones that actually do work). Probe steps 1-2 are not included.
    """
    commit_sha = _resolve_local_sha(working_tree)
    step_durations_s: dict[str, float] = {}

    # 1. SSH reach.
    r = ssh.run(host, "echo bootstrap-ok", timeout_s=15)
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"ssh unreachable: {r.stderr.strip() or r.stdout.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
        )

    # 2. Docker daemon reach.
    r = ssh.run(host, "docker ps --format '{{.Names}}' 2>&1", timeout_s=15)
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"docker daemon not responding: {r.stderr.strip() or r.stdout.strip()}",
            commit_sha=commit_sha,
        )

    # 3. Wipe.
    t0 = _time_step()
    r = ssh.run(host, f"rm -rf {host.isaaclab_path}", timeout_s=60)
    step_durations_s["wipe"] = _time_step() - t0
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"failed to wipe {host.isaaclab_path!r}: {r.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 4. Rsync.
    t0 = _time_step()
    rr = rsync.push(host, working_tree, host.isaaclab_path)
    step_durations_s["rsync"] = _time_step() - t0
    if rr.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"rsync push failed: {rr.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 5. Container start.
    t0 = _time_step()
    started = _container_start(host, ssh, timeout_s=build_timeout_s)
    step_durations_s["container_start"] = _time_step() - t0
    if not started:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"container.py start failed (timeout={build_timeout_s}s)",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 6. Container verify.
    t0 = _time_step()
    r = ssh.run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}",
        timeout_s=15,
    )
    step_durations_s["container_verify"] = _time_step() - t0
    status = r.stdout.strip()
    if r.exit_code != 0 or status != "running":
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"container {host.container_name!r} not running after start (status={status!r})",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    return BootstrapResult(
        host=host.host,
        ok=True,
        commit_sha=commit_sha,
        step_durations_s=step_durations_s,
    )


def bootstrap_fleet(
    fleet: Fleet,
    working_tree: Path,
    *,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    build_timeout_s: int = 1800,
    parallel: bool = True,
    verbose: bool = False,
) -> list[BootstrapResult]:
    """Bootstrap every host in ``fleet``.

    Args:
        fleet: Loaded fleet (via :func:`~tools.odin.asgard.fleet.load_fleet`).
        working_tree: Controller-side IsaacLab path to push.
        ssh: SSH runner shared across hosts.
        rsync: Rsync runner shared across hosts.
        build_timeout_s: Per-host ``container.py start`` timeout [s].
        parallel: When ``True`` (default), use a thread pool with
            ``max_workers = len(fleet.hosts)`` so hosts bootstrap
            concurrently. When ``False``, bootstrap sequentially —
            useful when shared network bandwidth would be saturated
            by simultaneous rsyncs.
        verbose: When ``True``, print a per-host summary line as each
            host finishes.

    Returns:
        :class:`BootstrapResult` list, one per host, in fleet order.
    """
    if parallel and len(fleet.hosts) > 1:
        with _cf.ThreadPoolExecutor(max_workers=len(fleet.hosts)) as pool:
            futures = [
                pool.submit(
                    bootstrap_valkyrie,
                    h,
                    working_tree,
                    ssh=ssh,
                    rsync=rsync,
                    build_timeout_s=build_timeout_s,
                )
                for h in fleet.hosts
            ]
            results = [f.result() for f in futures]
    else:
        results = [
            bootstrap_valkyrie(
                h,
                working_tree,
                ssh=ssh,
                rsync=rsync,
                build_timeout_s=build_timeout_s,
            )
            for h in fleet.hosts
        ]

    if verbose:
        for r in results:
            status = "ok" if r.ok else f"FAILED: {r.message}"
            print(f"[{r.host}] {status}")

    return results
```

- [ ] **Step 4: Run tests to verify the 9 single-host tests pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap.py -v --confcutdir=tools/odin
```

Expected: 9 PASS.

- [ ] **Step 5: Run pre-commit + commit**

```bash
./isaaclab.sh -f
# Re-run until clean if hooks auto-fix.
git add tools/odin/asgard/bootstrap.py tools/odin/tests/test_asgard_bootstrap.py
git commit -m "Add bootstrap_valkyrie per-host pipeline"
```

Subject is 44 chars.

---

### Task 3: `bootstrap_fleet` parallelism tests

**Files:**
- Modify: `tools/odin/tests/test_asgard_bootstrap.py`

- [ ] **Step 1: Append the fleet-level tests**

Append to `tools/odin/tests/test_asgard_bootstrap.py`:

```python
def test_bootstrap_fleet_returns_results_in_fleet_order(tmp_path: Path):
    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v2", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v3", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )
    ssh = _FakeSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()
    results = bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=False)
    assert [r.host for r in results] == ["v1", "v2", "v3"]
    assert all(r.ok for r in results)


def test_bootstrap_fleet_mixed_outcome(tmp_path: Path):
    """One host reaches SSH fine, another fails; both appear in the result list."""
    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v-good", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v-bad", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )

    # A per-host SSH fake: v-bad's first probe fails; v-good otherwise normal.
    good_ssh = _FakeSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    bad_ssh = _FakeSSH(replies={"echo bootstrap-ok": 255}, reply_stderr={"echo bootstrap-ok": "conn refused"})

    # Wrap both with a routing SSH that dispatches on host.host.
    @dataclass
    class _RoutingSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            inner = good_ssh if host.host == "v-good" else bad_ssh
            return inner.run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    results = bootstrap_fleet(
        fleet, tmp_path, ssh=_RoutingSSH(), rsync=_FakeRsync(), parallel=False
    )
    assert len(results) == 2
    good = next(r for r in results if r.host == "v-good")
    bad = next(r for r in results if r.host == "v-bad")
    assert good.ok is True
    assert bad.ok is False
    assert "ssh unreachable" in bad.message


def test_bootstrap_fleet_parallel_runs_concurrently(tmp_path: Path):
    """With 3 hosts and parallel=True, wall time ≈ max(per-host), not sum."""
    import time as _time_mod

    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p")
            for i in (1, 2, 3)
        ],
    )

    # SSH fake that sleeps 100 ms on container.py start to simulate slow hosts.
    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "container.py start" in cmd:
                _time_mod.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()

    t0 = _time_mod.perf_counter()
    results = bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=True)
    elapsed = _time_mod.perf_counter() - t0

    assert all(r.ok for r in results)
    # Serial would be ≥ 3 * 0.1 = 0.3 s. Parallel should be < 0.25 s.
    assert elapsed < 0.25, f"parallel=True elapsed={elapsed:.3f}s (expected <0.25)"


def test_bootstrap_fleet_sequential_adds_up(tmp_path: Path):
    """With parallel=False, wall time grows linearly with host count."""
    import time as _time_mod

    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p")
            for i in (1, 2, 3)
        ],
    )

    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "container.py start" in cmd:
                _time_mod.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()

    t0 = _time_mod.perf_counter()
    bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=False)
    elapsed = _time_mod.perf_counter() - t0

    # Serial wall time ≥ 3 * 0.1 s — allow loose upper bound for scheduler noise.
    assert elapsed >= 0.28, f"parallel=False elapsed={elapsed:.3f}s (expected >=0.28)"


def test_bootstrap_fleet_verbose_prints_per_host(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host="v-only", ssh_user="u", ssh_key=None, isaaclab_path="/p")],
    )
    ssh = _FakeSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()
    bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=False, verbose=True)
    out = capsys.readouterr().out
    assert "v-only" in out
    assert "ok" in out
```

- [ ] **Step 2: Run tests to verify all 14 (9 + 5 new) pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap.py -v --confcutdir=tools/odin
```

Expected: 14 PASS.

- [ ] **Step 3: Pre-commit + commit**

```bash
./isaaclab.sh -f
# Re-run until clean.
git add tools/odin/tests/test_asgard_bootstrap.py
git commit -m "Add bootstrap_fleet parallel driver + tests"
```

Subject is 44 chars.

---

### Task 4: `odin-bootstrap` CLI

**Files:**
- Create: `tools/odin/asgard/bootstrap_cli.py`
- Create: `tools/odin/tests/test_asgard_bootstrap_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Create `tools/odin/tests/test_asgard_bootstrap_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the odin-bootstrap CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.bootstrap import BootstrapResult
from tools.odin.asgard.bootstrap_cli import main, parse_args


def _write_fleet_yaml(tmp_path: Path) -> Path:
    content = """\
fleet_name: test
default_ssh_user: odin
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: v1.internal
  - host: v2.internal
"""
    path = tmp_path / "fleet.yaml"
    path.write_text(content)
    return path


def test_parse_args_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["--fleet", str(fleet_path)])
    assert args.fleet == fleet_path
    assert args.build_timeout == 1800
    assert args.sequential is False
    assert args.verbose is False


def test_parse_args_all_flags(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(
        [
            "--fleet", str(fleet_path),
            "--build-timeout", "3600",
            "--sequential",
            "--verbose",
        ]
    )
    assert args.build_timeout == 3600
    assert args.sequential is True
    assert args.verbose is True


def test_main_exit_zero_when_all_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        return [
            BootstrapResult(host="v1.internal", ok=True),
            BootstrapResult(host="v2.internal", ok=True),
        ]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    exit_code = main(["--fleet", str(fleet_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "bootstrap complete: 2/2 hosts ok" in out


def test_main_exit_one_when_any_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        return [
            BootstrapResult(host="v1.internal", ok=True),
            BootstrapResult(host="v2.internal", ok=False, message="ssh unreachable: conn refused"),
        ]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    exit_code = main(["--fleet", str(fleet_path)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "bootstrap complete: 1/2 hosts ok" in out
    assert "v2.internal" in out
    assert "ssh unreachable" in out


def test_main_sequential_flag_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fleet_path = _write_fleet_yaml(tmp_path)
    recorded: dict = {}

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        recorded["parallel"] = parallel
        return [BootstrapResult(host="v1.internal", ok=True), BootstrapResult(host="v2.internal", ok=True)]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    main(["--fleet", str(fleet_path), "--sequential"])
    assert recorded["parallel"] is False


def test_main_build_timeout_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fleet_path = _write_fleet_yaml(tmp_path)
    recorded: dict = {}

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        recorded["build_timeout_s"] = build_timeout_s
        return [BootstrapResult(host="v1.internal", ok=True), BootstrapResult(host="v2.internal", ok=True)]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    main(["--fleet", str(fleet_path), "--build-timeout", "3600"])
    assert recorded["build_timeout_s"] == 3600
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap_cli.py -v --confcutdir=tools/odin
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tools.odin.asgard.bootstrap_cli'`.

- [ ] **Step 3: Implement the CLI**

Create `tools/odin/asgard/bootstrap_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-bootstrap — bring a fresh fleet to T3.1-preflight-ready state.

Usage::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py \\
        --fleet fleet.yaml \\
        [--build-timeout 1800] \\
        [--sequential] \\
        [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.bootstrap import bootstrap_fleet
from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the bootstrap CLI args. Factored out for unit testing."""
    parser = argparse.ArgumentParser(
        prog="odin-bootstrap",
        description=(
            "Bring a fresh Odin fleet to T3.1-preflight-ready state: wipe + "
            "rsync the working tree + start the isaac-lab-base container on "
            "every host. Idempotent by design (always wipe + re-rsync)."
        ),
    )
    parser.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    parser.add_argument(
        "--build-timeout",
        type=int,
        default=1800,
        help=(
            "Per-host timeout [s] for `./docker/container.py start`. "
            "Default 1800 (30 min) covers a first-time docker image build."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Bootstrap hosts one at a time instead of the default parallel "
            "(one thread per host). Useful when shared network bandwidth "
            "would be saturated by simultaneous rsyncs."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a per-host ok/fail summary line as each host finishes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Bootstrap the fleet; return 0 iff every host reported ok."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    fleet = load_fleet(args.fleet)

    results = bootstrap_fleet(
        fleet,
        Path.cwd(),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
        build_timeout_s=args.build_timeout,
        parallel=not args.sequential,
        verbose=args.verbose,
    )

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"bootstrap complete: {ok_count}/{total} hosts ok")
    if ok_count < total:
        for r in results:
            if not r.ok:
                print(f"  {r.host}: {r.message}")
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap_cli.py -v --confcutdir=tools/odin
```

Expected: 6 PASS.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
# Re-run until clean.
git add tools/odin/asgard/bootstrap_cli.py tools/odin/tests/test_asgard_bootstrap_cli.py
git commit -m "Add odin-bootstrap CLI wrapper"
```

Subject is 30 chars.

---

### Task 5: Public re-exports in `__init__.py`

**Files:**
- Modify: `tools/odin/asgard/__init__.py`

- [ ] **Step 1: Read current `__init__.py`**

```bash
cat tools/odin/asgard/__init__.py
```

Expected: existing re-exports for `Fleet`, `ValkyrieConfig`, `load_fleet`, `DispatchOptions`, `run_dispatch`, `resolve_dispatch_dir`, `JobEntry`, `FailureInfo`, `build_queue_from_env_lists`, `DispatchState`, `FleetSnapshot`, `read_dispatch_state`, `write_dispatch_state`, `reset_in_flight_to_pending`, `SCHEMA_VERSION`, `RsyncResult`, `RsyncRunner`, `ShellRsyncRunner`, `ShellSSHRunner`, `SSHResult`, `SSHRunner`, `preflight_valkyrie`, `PreflightResult`, `provision_valkyrie`, `ProvisionResult`, `ValkyrieWorker`, `WorkerOptions`, `StateEvent`.

- [ ] **Step 2: Add bootstrap symbols to the imports + `__all__`**

In `tools/odin/asgard/__init__.py`, find the section that imports from existing submodules and add a new import line for bootstrap after the `from tools.odin.asgard.provisioner import ...` line:

```python
from tools.odin.asgard.bootstrap import BootstrapResult, bootstrap_fleet, bootstrap_valkyrie
```

In the `__all__` list, insert the three new symbols in alphabetical position. Example if the current `__all__` is sorted: add `"BootstrapResult"`, `"bootstrap_fleet"`, `"bootstrap_valkyrie"` in the correct places. If the list is unsorted, append the three at the end.

- [ ] **Step 3: Verify the imports resolve**

```bash
./isaaclab.sh -p -c "from tools.odin.asgard import BootstrapResult, bootstrap_valkyrie, bootstrap_fleet; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Run the asgard test suite to confirm nothing else broke**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ -v --confcutdir=tools/odin -k "not slow"
```

Expected: all tests pass (existing ones + Tasks 1-4's new tests).

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/__init__.py
git commit -m "Export bootstrap symbols from asgard package"
```

Subject is 43 chars.

---

### Task 6: README — "Bootstrapping a fresh fleet" section

**Files:**
- Modify: `tools/odin/README.md`

- [ ] **Step 1: Locate the right insertion point**

```bash
grep -n "^## " tools/odin/README.md
```

Find the existing `## Dispatching across a fleet (T3.1 — Asgard)` section. Insert the new bootstrap section directly before it (bootstrap is a prerequisite for dispatch on fresh fleets, so it reads in order).

- [ ] **Step 2: Insert the new section**

Before the line `## Dispatching across a fleet (T3.1 — Asgard)` in `tools/odin/README.md`, insert:

```markdown
## Bootstrapping a fresh fleet

Fresh Valkyries — machines with only SSH, Docker, and a GPU — need to be
brought to T3.1-preflight-ready state before `odin-dispatch` will run
against them. `odin-bootstrap` handles this:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py \
    --fleet fleet.yaml \
    [--build-timeout 1800] \
    [--sequential] \
    [--verbose]
```

For each host in `fleet.yaml`:

1. SSH reach + Docker daemon probes (fast).
2. `rm -rf {isaaclab_path}` (always — keeps re-runs idempotent).
3. `rsync` the controller's working tree to `{isaaclab_path}`.
4. `./docker/container.py start` with `--build-timeout` (default 1800 s
   = 30 min — covers a first-time docker image build).
5. `docker inspect` must report the container as `running`.

Hosts bootstrap in parallel by default (one thread per host); pass
`--sequential` if shared network bandwidth can't support simultaneous
rsyncs. Exit code is `0` iff every host succeeded.

After `odin-bootstrap` returns green, the T3.1 dispatch path works:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --seeds 42
```

Re-bootstrapping wipes and re-rsyncs — useful after long host idle (image
cache evicted), container drift, or as a belt-and-braces maintenance
pass. Don't re-bootstrap mid-dispatch: stop the dispatch first.

```

- [ ] **Step 3: Verify rendered layout (spot check)**

```bash
grep -n "^## Bootstrapping\|^## Dispatching" tools/odin/README.md
```

Expected: the new section header immediately precedes the existing dispatch header.

- [ ] **Step 4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/README.md
git commit -m "Document odin-bootstrap in README"
```

Subject is 31 chars.

---

### Task 7: Full-suite sweep + architecture doc update

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 1: Run the complete test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ -v --confcutdir=tools/odin
```

Expected: all existing tests plus the 21 new bootstrap/CLI tests pass. Rough expected count: ~190 passed + 1 skipped.

- [ ] **Step 2: Run pre-commit one final time**

```bash
./isaaclab.sh -f
```

Expected: clean.

- [ ] **Step 3: Update `docs/odin/architecture.md`**

Find the existing T3 row in §6's task map:

```
| T3 | Distributed dispatcher (Layer 3) + Asgard | `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md` | 🟡 |
```

Update the spec column to list both specs (T3.1 dispatch + bootstrap) — the T3 column is the right home for bootstrap since bootstrap extends the Asgard layer. Replace with:

```
| T3 | Distributed dispatcher (Layer 3) + Asgard | T3.1 `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md`; bootstrap `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md` | 🟡 |
```

Bump the "Last updated" line:

```
**Last updated:** 2026-04-23 (end of T3 bootstrap)
```

Add a change-log entry to §9:

```
| 2026-04-23 | Odin bootstrap delivered. New `tools/odin/asgard/bootstrap.py` + `bootstrap_cli.py` (spec: `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md`) closes a T3.1 gap: fresh Valkyries (just SSH + Docker + GPU) can now be brought to preflight-ready state with a single `odin-bootstrap --fleet fleet.yaml` invocation. Pipeline: ssh reach → docker daemon probe → `rm -rf {isaaclab_path}` → rsync working tree → `./docker/container.py start` (30-min timeout, configurable) → `docker inspect` verify. Parallel per-host by default (thread pool); `--sequential` opt-out. Provisioner tweak: `_container_start` gains a `timeout_s` keyword so the warm-path 300 s default stays while bootstrap passes 1800 s. Zero changes to T3.1's `run_dispatch` / preflight / worker. 13 unit tests for `bootstrap_valkyrie` + `bootstrap_fleet`, 6 CLI tests. | Odin T3 bootstrap |
```

- [ ] **Step 4: Commit doc update**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin bootstrap complete in architecture reference"
```

Subject is 54 chars — exceeds 50. Use `Mark Odin bootstrap complete in architecture` (46 chars) instead.

---

## Summary of verification criteria

After all 7 tasks, the following must hold:

- `./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ -v --confcutdir=tools/odin` → all pass (~190 + the 22 new bootstrap/CLI tests: 14 bootstrap + 6 CLI + 2 provisioner).
- `./isaaclab.sh -f` → clean.
- `tools/odin/asgard/bootstrap.py` + `bootstrap_cli.py` exist; re-exports in `__init__.py`.
- `odin-bootstrap --fleet fleet.yaml --help` runs and shows the four flags (`--fleet`, `--build-timeout`, `--sequential`, `--verbose`).
- Architecture doc reflects bootstrap; README has the new section.

## What comes next (out of scope for this plan)

- **Actually bootstrap** the two machines (10.176.221.98, 10.63.172.46). Runs `odin-bootstrap --fleet fleet.yaml --verbose` — expect ~20-30 min for the first-time docker image build per host.
- **Run the deferred T4.1 real-fleet validation** (5-task × 3-seed × 2-backend dispatch) after bootstrap succeeds.
- Any gaps that surface during the real run are folded into follow-up fix commits on this branch.
