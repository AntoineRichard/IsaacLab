# Odin Live Job Tail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Tab A show the last 50 lines of a running Odin job's Hugin stdout while the job is still in flight.

**Architecture:** Hugin writes each phase's stdout/stderr directly to bundle log files during the child process run. The dashboard data layer tails the remote bundle-side `training.stdout.log`, falling back to `startup.stdout.log`, through a short SSH + `docker exec` call. Tab A mirrors the existing failed-row ssh-tail pattern with a running-row toggle, a refresh button, and per-run store state.

**Tech Stack:** Python 3.10+, Dash pattern-matching callbacks, `subprocess.run`, `ssh`, `docker exec`, pytest, pre-commit.

---

## Execution Rules

- Work in `/home/antoiner/Documents/IsaacLab` on branch `antoiner/feat/odin`.
- Use a fresh implementation subagent per task after plan approval.
- After each task implementation, run two fresh review subagents before committing:
  - Spec-compliance review: verify the diff satisfies this plan and `docs/superpowers/specs/2026-04-30-odin-live-job-tail-design.md`.
  - Code-quality review: verify maintainability, test coverage, edge cases, and local conventions.
- Use TDD inside every task: write failing tests first, run the exact new test IDs and see them fail, then implement.
- Run tests one at a time:

```bash
PYTHONPATH=. python3 -m pytest <test_id> -v --tb=short --noconftest -p no:cacheprovider
```

- Run pre-commit before each commit, limited to touched files:

```bash
pre-commit run --files <files>
```

- Stage only the files touched by the task.
- If pre-commit modifies unrelated files, restore only those unrelated files before committing.
- Use conventional commits. Subjects must be imperative, capitalized, no trailing period, and at most 50 characters.
- Do not add AI attribution or co-authorship lines.

---

## File Structure

### Hugin Streaming

- Modify `tools/odin/hugin/run.py`
  - Remove the capture-output failure-only log write path.
  - Add `_with_unbuffered_python(cmd)` to inject `-u` for Python-like commands.
  - Open `<phase>.stdout.log` and `<phase>.stderr.log` before launching each phase and pass those file handles to `_subprocess_run`.
- Modify `tools/odin/tests/test_hugin.py`
  - Update existing fake subprocess helpers to write through provided `stdout`/`stderr` file handles.
  - Add streaming, full-log, no-truncation, and `-u` injection regression tests.

### Running Tail Data Layer

- Modify `tools/odin/valhalla/dashboard/data.py`
  - Add `_subprocess_run = subprocess.run` for monkeypatchable SSH tests.
  - Add `DataLayer.read_running_job_tail(...) -> list[str]`.
  - Add `DataLayer.read_running_job_tail_payload(...) -> dict[str, Any]` for UI metadata: `{"source": "training.stdout.log" | "startup.stdout.log" | None, "lines": [...]}`.
  - Add small private helpers to build the SSH argv, build the remote `docker exec` command, and parse the source marker.
- Create `tools/odin/valhalla/dashboard/tests/test_data_running_tail.py`
  - Unit-test SSH command construction, fallback ordering, empty/missing logs, line cap, SSH failure warning, and lossy decode.

### Tab A UI

- Modify `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`
  - Add stores `tab-a-running-tail-shown` and `tab-a-running-tail-store`.
- Modify `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
  - Add `running_tail_shown` and `running_tail_store` inputs to `render_jobs_section()` and `render_jobs_rows()`.
  - Render a running-row `dcc.Button` with id `{type: "tab-a-running-tail-toggle", run_id: <id>}`.
  - Add `_expand_running_row(job, tail_entry)` using the existing expand row styling.
- Modify `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`
  - Include running-tail stores in `_update_jobs`.
  - Add a visibility toggle callback.
  - Add a fetch/refresh callback that finds the job host from `dispatch.json` and calls the data layer.
  - Add pure helpers for callback tests.
- Modify `tools/odin/valhalla/dashboard/assets/style.css`
  - Reuse existing expand/ssh-tail styles and add small running-tail aliases.
- Modify `tools/odin/valhalla/dashboard/tests/test_app.py`
  - Assert the two new stores exist in Tab A layout.
- Modify `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`
  - Add callback helper round-trip tests.
- Create `tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py`
  - Unit-test running-row button rendering, non-running exclusion, expand row rendering, and filename marker display.

---

## Task 1: Stream Hugin Phase Logs

**Files:**
- Modify: `tools/odin/hugin/run.py`
- Modify: `tools/odin/tests/test_hugin.py`

- [ ] **Step 1: Add failing Hugin streaming tests**

Add these helpers and tests to `tools/odin/tests/test_hugin.py`. Keep the existing tests, but update their fakes in the same file to write through `stdout` and `stderr` when those handles are passed.

```python
class _Completed:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


def _write_output_json_from_cmd(cmd: list[str]) -> str:
    out_idx = cmd.index("--schema_v1_output") + 1
    out_path = cmd[out_idx]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write('{"schema_version": "1.0", "fake": true}\n')
    return out_path


def test_run_phase_streams_stdout_to_log_during_run(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    output_json = bundle / "training.json"
    observed_during_run = []

    def _streaming_run(cmd, *args, **kwargs):
        _write_output_json_from_cmd(cmd)
        stdout = kwargs["stdout"]
        stdout.write(b"iter 1 reward=0.1\n")
        stdout.flush()
        observed_during_run.append(open(stdout.name, "rb").read())
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _streaming_run)

    phase = hugin_run._run_phase(
        ["python.sh", "train.py", "--schema_v1_output", str(output_json)],
        str(bundle),
        "training",
        str(output_json),
    )

    assert phase.status == "completed"
    assert observed_during_run == [b"iter 1 reward=0.1\n"]


def test_run_phase_writes_full_stdout_on_completed(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    output_json = bundle / "training.json"

    def _streaming_run(cmd, *args, **kwargs):
        _write_output_json_from_cmd(cmd)
        kwargs["stdout"].write(b"first\n")
        kwargs["stdout"].write(b"second\n")
        kwargs["stderr"].write(b"warning\n")
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _streaming_run)

    phase = hugin_run._run_phase(
        ["python.sh", "train.py", "--schema_v1_output", str(output_json)],
        str(bundle),
        "training",
        str(output_json),
    )

    assert phase.status == "completed"
    assert (bundle / "logs" / "training.stdout.log").read_bytes() == b"first\nsecond\n"
    assert (bundle / "logs" / "training.stderr.log").read_bytes() == b"warning\n"


def test_run_phase_no_tail_truncation_on_failure(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    output_json = bundle / "training.json"
    full_stdout = b"".join(f"line-{i}\n".encode() for i in range(2000))
    full_stderr = b"".join(f"err-{i}\n".encode() for i in range(2000))

    def _failing_run(cmd, *args, **kwargs):
        kwargs["stdout"].write(full_stdout)
        kwargs["stderr"].write(full_stderr)
        return _Completed(7)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _failing_run)

    phase = hugin_run._run_phase(
        ["python.sh", "train.py", "--schema_v1_output", str(output_json)],
        str(bundle),
        "training",
        str(output_json),
    )

    assert phase.status == "failed"
    assert phase.exit_code == 7
    assert (bundle / "logs" / "training.stdout.log").read_bytes() == full_stdout
    assert (bundle / "logs" / "training.stderr.log").read_bytes() == full_stderr


def test_run_phase_injects_dash_u_for_python_child(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    output_json = bundle / "training.json"
    captured_cmds = []

    def _recording_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _recording_run)

    hugin_run._run_phase(
        ["python.sh", "train.py", "--schema_v1_output", str(output_json)],
        str(bundle),
        "training",
        str(output_json),
    )

    assert captured_cmds == [["python.sh", "-u", "train.py", "--schema_v1_output", str(output_json)]]


def test_run_phase_injects_dash_u_for_isaaclab_sh_python_mode(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    output_json = bundle / "training.json"
    captured_cmds = []

    def _recording_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _recording_run)

    hugin_run._run_phase(
        ["isaaclab.sh", "-p", "train.py", "--schema_v1_output", str(output_json)],
        str(bundle),
        "training",
        str(output_json),
    )

    assert captured_cmds == [["isaaclab.sh", "-p", "-u", "train.py", "--schema_v1_output", str(output_json)]]


def test_run_phase_does_not_inject_dash_u_for_non_python(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    output_json = bundle / "probe.json"
    captured_cmds = []

    def _recording_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _recording_run)

    hugin_run._run_phase(
        ["nvidia-smi", "--schema_v1_output", str(output_json)],
        str(bundle),
        "probe",
        str(output_json),
    )

    assert captured_cmds == [["nvidia-smi", "--schema_v1_output", str(output_json)]]
```

Update the existing `_fake_run_factory()` so it writes logs through file handles:

```python
def _fake_run_factory():
    """Return a stub that pretends to write startup.json/training.json and
    records every command it was called with for later assertions."""

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        _write_output_json_from_cmd(cmd)
        if kwargs.get("stdout") is not None:
            kwargs["stdout"].write(b"fake stdout")
        if kwargs.get("stderr") is not None:
            kwargs["stderr"].write(b"fake stderr")
        return _Completed(0)

    _fake_run.captured_cmds = captured_cmds
    return _fake_run
```

Update `test_hugin_failure_path_writes_logs()` so `_failing_run()` writes through the provided handles:

```python
def _failing_run(cmd, *args, **kwargs):
    kwargs["stdout"].write(b"partial stdout")
    kwargs["stderr"].write(b"traceback...")
    return _Completed(7)
```

- [ ] **Step 2: Run each new Hugin test and verify it fails**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_streams_stdout_to_log_during_run -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_writes_full_stdout_on_completed -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_no_tail_truncation_on_failure -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_injects_dash_u_for_python_child -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_injects_dash_u_for_isaaclab_sh_python_mode -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_does_not_inject_dash_u_for_non_python -v --tb=short --noconftest -p no:cacheprovider
```

Expected: at least the streaming/full-log tests fail because `_run_phase()` still calls `_subprocess_run(..., capture_output=True)` and only writes failure logs from captured buffers. The `-u` injection tests fail because the command is passed through unchanged.

- [ ] **Step 3: Implement streamed log files in Hugin**

In `tools/odin/hugin/run.py`, remove `from tools.odin.common.log_tail import tail_bytes` and replace `_run_phase()` with this implementation plus the helper:

```python
def _with_unbuffered_python(cmd: list[str]) -> list[str]:
    """Return ``cmd`` with ``-u`` inserted for Python child processes.

    Hugin redirects child stdout/stderr to files so CPython would otherwise
    block-buffer stdout. For Python launchers, ``-u`` keeps per-iteration log
    lines visible in the bundle within the child's own flush behavior.
    """
    if not cmd:
        return []
    out = list(cmd)
    exe = os.path.basename(out[0])
    is_python = exe in {"python", "python3", "python.sh"} or exe.endswith(("python", "python3", "python.sh"))
    if is_python:
        if len(out) == 1 or out[1] != "-u":
            out.insert(1, "-u")
        return out
    if exe == "isaaclab.sh" and len(out) >= 2 and out[1] == "-p":
        if len(out) == 2 or out[2] != "-u":
            out.insert(2, "-u")
        return out
    return out


def _run_phase(cmd: list[str], bundle_dir: str, phase_name: str, output_json: str) -> ManifestPhase:
    """Run one subprocess phase; stream stdout/stderr to bundle log files.

    Defines "completed" as: returncode == 0 AND ``output_json`` exists. A
    silent-exit-0 (subprocess exits 0 but writes no output) is a known
    failure mode for Isaac Sim crashes — promote it to ``status="failed"``
    with a derived non-zero exit code so the worker's classifier and the
    aggregator both pick it up.
    """
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stdout_path = os.path.join(logs_dir, f"{phase_name}.stdout.log")
    stderr_path = os.path.join(logs_dir, f"{phase_name}.stderr.log")
    start = datetime.now(timezone.utc)
    with open(stdout_path, "wb") as stdout_fh, open(stderr_path, "wb") as stderr_fh:
        completed = _subprocess_run(_with_unbuffered_python(cmd), stdout=stdout_fh, stderr=stderr_fh)
    end = datetime.now(timezone.utc)
    duration_s = (end - start).total_seconds()
    output_exists = os.path.exists(output_json)
    if completed.returncode != 0 or not output_exists:
        status = "failed"
        exit_code = completed.returncode or 1
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

- [ ] **Step 4: Run each Hugin test and verify it passes**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_streams_stdout_to_log_during_run -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_writes_full_stdout_on_completed -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_no_tail_truncation_on_failure -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_injects_dash_u_for_python_child -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_injects_dash_u_for_isaaclab_sh_python_mode -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_run_phase_does_not_inject_dash_u_for_non_python -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_hugin_happy_path -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_hugin_failure_path_writes_logs -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/tests/test_hugin.py::test_hugin_silent_exit_zero_no_output_marks_failed -v --tb=short --noconftest -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 5: Run pre-commit for Hugin files**

```bash
pre-commit run --files tools/odin/hugin/run.py tools/odin/tests/test_hugin.py
```

Expected: pass. If formatting changes these two files, review the diff and rerun the command.

- [ ] **Step 6: Run two review gates for Task 1**

Spec-compliance review prompt:

```text
Review Task 1 only. Compare the diff in tools/odin/hugin/run.py and tools/odin/tests/test_hugin.py against docs/superpowers/specs/2026-04-30-odin-live-job-tail-design.md Part A and docs/superpowers/plans/2026-04-30-odin-live-job-tail.md Task 1. Report only concrete spec gaps, test gaps, or regressions.
```

Code-quality review prompt:

```text
Review Task 1 only for code quality. Focus on subprocess semantics, log file behavior, -u insertion safety, test determinism, and IsaacLab conventions. Report concrete issues with file/line references.
```

Address review findings before committing.

- [ ] **Step 7: Commit Task 1**

```bash
git add tools/odin/hugin/run.py tools/odin/tests/test_hugin.py
git commit -m "Stream Hugin phase logs" \
  -m "Write phase stdout and stderr directly to bundle log files." \
  -m "Keep silent-exit handling and unbuffered Python launch coverage."
```

---

## Task 2: Add Running Tail DataLayer Reader

**Files:**
- Modify: `tools/odin/valhalla/dashboard/data.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_data_running_tail.py`

- [ ] **Step 1: Write failing DataLayer tests**

Create `tools/odin/valhalla/dashboard/tests/test_data_running_tail.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DataLayer.read_running_job_tail."""

from __future__ import annotations

import subprocess

from tools.odin.valhalla.dashboard import data as data_mod
from tools.odin.valhalla.dashboard.data import DataLayer


class _Result:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_run(monkeypatch, result: _Result, calls: list[list[str]]):
    def _fake_run(argv, *, capture_output, timeout, check):
        calls.append(list(argv))
        assert capture_output is True
        assert timeout == 10
        assert check is False
        return result

    monkeypatch.setattr(data_mod, "_subprocess_run", _fake_run)


def _marked(source: str, lines: list[str]) -> bytes:
    return (f"__odin_tail_source__:{source}\n" + "\n".join(lines) + "\n").encode()


def test_read_running_tail_returns_lines_from_training_log_when_present(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    expected = [f"line-{i}" for i in range(50)]
    _install_fake_run(monkeypatch, _Result(stdout=_marked("training.stdout.log", expected)), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-1", host="v1")

    assert lines == expected
    assert calls[0][0] == "ssh"
    assert "-o" in calls[0]
    assert "StrictHostKeyChecking=accept-new" in calls[0]
    assert "ConnectTimeout=5" in calls[0]
    assert "BatchMode=yes" in calls[0]
    assert calls[0][-2] == "horde@v1"
    assert "docker exec isaac-lab-base bash -c" in calls[0][-1]
    assert "training.stdout.log" in calls[0][-1]


def test_read_running_tail_falls_back_to_startup_when_training_empty(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, _Result(stdout=_marked("startup.stdout.log", ["startup ready"])), calls)

    entry = DataLayer(tmp_path).read_running_job_tail_payload("20260430-110509", "run-2", host="v1")

    assert entry == {"source": "startup.stdout.log", "lines": ["startup ready"]}
    remote_cmd = calls[0][-1]
    assert remote_cmd.index("training.stdout.log") < remote_cmd.index("startup.stdout.log")


def test_read_running_tail_returns_empty_list_when_no_logs_yet(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, _Result(stdout=b""), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-3", host="v1")

    assert lines == []


def test_read_running_tail_caps_at_n_lines(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    all_lines = [f"line-{i}" for i in range(200)]
    _install_fake_run(monkeypatch, _Result(stdout=_marked("training.stdout.log", all_lines)), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-4", host="v1", n=50)

    assert lines == all_lines[-50:]


def test_read_running_tail_uses_custom_ssh_key_user_container_and_line_count(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    key = tmp_path / "id_ed25519"
    key.write_text("fake")
    _install_fake_run(monkeypatch, _Result(stdout=_marked("training.stdout.log", ["x"])), calls)

    DataLayer(tmp_path).read_running_job_tail(
        "20260430-110509",
        "run-5",
        host="10.0.0.5",
        ssh_user="odin",
        ssh_key=key,
        container_name="custom-container",
        n=17,
    )

    argv = calls[0]
    assert "-i" in argv
    assert argv[argv.index("-i") + 1] == str(key)
    assert argv[-2] == "odin@10.0.0.5"
    assert "docker exec custom-container bash -c" in argv[-1]
    assert "tail -n 17" in argv[-1]


def test_read_running_tail_ssh_failure_returns_empty_with_warning(tmp_path, monkeypatch, capsys):
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, _Result(stderr=b"connection timed out", returncode=255), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-6", host="v1")

    assert lines == []
    captured = capsys.readouterr()
    assert "[WARNING] read_running_job_tail 20260430-110509/run-6" in captured.err
    assert "connection timed out" in captured.err


def test_read_running_tail_timeout_returns_empty_with_warning(tmp_path, monkeypatch, capsys):
    def _timeout(argv, *, capture_output, timeout, check):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(data_mod, "_subprocess_run", _timeout)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-7", host="v1")

    assert lines == []
    captured = capsys.readouterr()
    assert "TimeoutExpired" in captured.err


def test_read_running_tail_handles_binary_garbage_gracefully(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    payload = b"__odin_tail_source__:training.stdout.log\nok\nbad:\xff\n"
    _install_fake_run(monkeypatch, _Result(stdout=payload), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-8", host="v1")

    assert lines[0] == "ok"
    assert lines[1].startswith("bad:")
```

- [ ] **Step 2: Run each DataLayer test and verify it fails**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_returns_lines_from_training_log_when_present -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_falls_back_to_startup_when_training_empty -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_returns_empty_list_when_no_logs_yet -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_caps_at_n_lines -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_uses_custom_ssh_key_user_container_and_line_count -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_ssh_failure_returns_empty_with_warning -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_timeout_returns_empty_with_warning -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_handles_binary_garbage_gracefully -v --tb=short --noconftest -p no:cacheprovider
```

Expected: fail because `DataLayer` has no running-tail methods and no `_subprocess_run` alias.

- [ ] **Step 3: Implement the DataLayer running-tail reader**

In `tools/odin/valhalla/dashboard/data.py`, add imports:

```python
import shlex
import subprocess
import sys
```

Add module constants and the monkeypatchable subprocess alias near `_DISPATCH_ID_RE`:

```python
_REMOTE_ODIN_RUNS_ROOT = "/workspace/isaaclab/odin_runs"
_RUNNING_TAIL_DEFAULT_LINES = 50
_RUNNING_TAIL_SOURCE_PREFIX = "__odin_tail_source__:"
_RUNNING_TAIL_TIMEOUT_S = 10
_subprocess_run = subprocess.run
```

Add these methods to `DataLayer` before `invalidate()`:

```python
    def read_running_job_tail(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        host: str,
        ssh_user: str = "horde",
        ssh_key: Path | None = None,
        container_name: str = "isaac-lab-base",
        n: int = _RUNNING_TAIL_DEFAULT_LINES,
    ) -> list[str]:
        """Return the last ``n`` lines of a running job's Hugin stdout.

        Tails ``training.stdout.log`` first, then falls back to
        ``startup.stdout.log`` when training has not emitted output yet.
        Returns an empty list when neither log exists, both logs are empty,
        or the short SSH call fails.
        """
        return self.read_running_job_tail_payload(
            dispatch_id,
            run_id,
            host=host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            container_name=container_name,
            n=n,
        )["lines"]

    def read_running_job_tail_payload(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        host: str,
        ssh_user: str = "horde",
        ssh_key: Path | None = None,
        container_name: str = "isaac-lab-base",
        n: int = _RUNNING_TAIL_DEFAULT_LINES,
    ) -> dict[str, Any]:
        """Return running-tail render data for a dashboard row.

        The public :meth:`read_running_job_tail` keeps the spec's ``list[str]``
        return shape. This payload variant also includes the selected source
        filename so the Tab A expand row can label ``training.stdout.log`` vs
        ``startup.stdout.log`` accurately.
        """
        bounded_n = max(1, int(n))
        ssh_cmd = _build_running_tail_ssh_cmd(run_id=run_id, container_name=container_name, n=bounded_n)
        argv = _build_running_tail_ssh_argv(host=host, ssh_user=ssh_user, ssh_key=ssh_key, ssh_cmd=ssh_cmd)
        try:
            result = _subprocess_run(
                argv,
                capture_output=True,
                timeout=_RUNNING_TAIL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _warn_running_tail(dispatch_id, run_id, f"{type(exc).__name__}: {exc}")
            return {"source": None, "lines": []}
        if result.returncode != 0:
            stderr = _decode_running_tail_bytes(result.stderr).strip()
            _warn_running_tail(dispatch_id, run_id, stderr or f"ssh exited {result.returncode}")
            return {"source": None, "lines": []}
        source, lines = _parse_running_tail_stdout(result.stdout, bounded_n)
        return {"source": source, "lines": lines}
```

Add these module-level helpers near `_summary_from_dispatch()`:

```python
def _build_running_tail_ssh_argv(
    *,
    host: str,
    ssh_user: str,
    ssh_key: Path | None,
    ssh_cmd: str,
) -> list[str]:
    argv = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
    ]
    if ssh_key is not None:
        argv += ["-i", str(ssh_key)]
    argv += [f"{ssh_user}@{host}", ssh_cmd]
    return argv


def _build_running_tail_ssh_cmd(*, run_id: str, container_name: str, n: int) -> str:
    logs_dir = shlex.quote(f"{_REMOTE_ODIN_RUNS_ROOT}/{run_id}/logs")
    inner = (
        f"base={logs_dir}; "
        "for name in training.stdout.log startup.stdout.log; do "
        'f="$base/$name"; '
        'if [ -s "$f" ]; then '
        f'printf "{_RUNNING_TAIL_SOURCE_PREFIX}%s\\n" "$name"; '
        f'tail -n {n} "$f"; '
        "exit 0; "
        "fi; "
        "done"
    )
    return f"docker exec {shlex.quote(container_name)} bash -c {shlex.quote(inner)}"


def _parse_running_tail_stdout(stdout: bytes | str | None, n: int) -> tuple[str | None, list[str]]:
    text = _decode_running_tail_bytes(stdout)
    lines = text.splitlines()
    source = None
    if lines and lines[0].startswith(_RUNNING_TAIL_SOURCE_PREFIX):
        source = lines[0][len(_RUNNING_TAIL_SOURCE_PREFIX) :] or None
        lines = lines[1:]
    return source, lines[-n:]


def _decode_running_tail_bytes(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _warn_running_tail(dispatch_id: str, run_id: str, message: str) -> None:
    print(f"[WARNING] read_running_job_tail {dispatch_id}/{run_id}: {message}", file=sys.stderr)
```

- [ ] **Step 4: Run each DataLayer test and verify it passes**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_returns_lines_from_training_log_when_present -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_falls_back_to_startup_when_training_empty -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_returns_empty_list_when_no_logs_yet -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_caps_at_n_lines -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_uses_custom_ssh_key_user_container_and_line_count -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_ssh_failure_returns_empty_with_warning -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_timeout_returns_empty_with_warning -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_data_running_tail.py::test_read_running_tail_handles_binary_garbage_gracefully -v --tb=short --noconftest -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 5: Run pre-commit for DataLayer files**

```bash
pre-commit run --files tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data_running_tail.py
```

Expected: pass. If formatting changes these two files, review the diff and rerun the command.

- [ ] **Step 6: Run two review gates for Task 2**

Spec-compliance review prompt:

```text
Review Task 2 only. Compare tools/odin/valhalla/dashboard/data.py and tools/odin/valhalla/dashboard/tests/test_data_running_tail.py against Part B of docs/superpowers/specs/2026-04-30-odin-live-job-tail-design.md and Task 2 in the implementation plan. Verify fallback order, SSH options, empty behavior, lossy decode, and warning behavior.
```

Code-quality review prompt:

```text
Review Task 2 only for code quality. Focus on shell quoting, subprocess API usage, warning behavior, test isolation, and whether the metadata payload helper preserves the required read_running_job_tail list[str] API.
```

Address review findings before committing.

- [ ] **Step 7: Commit Task 2**

```bash
git add tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data_running_tail.py
git commit -m "Add running job tail reader" \
  -m "Read running Hugin stdout through a short SSH docker-exec tail." \
  -m "Return empty on missing logs and expose source metadata for Tab A."
```

---

## Task 3: Wire Running Tail into Tab A

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`
- Modify: `tools/odin/valhalla/dashboard/assets/style.css`
- Modify: `tools/odin/valhalla/dashboard/tests/test_app.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py`

- [ ] **Step 1: Write failing Tab A running-tail render tests**

Create `tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Tab A running-job tail rendering."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import _expand_running_row, render_jobs_section


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


def _job(*, run_id="r", status="running", kind=None):
    job = {
        "run_id": run_id,
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-30T12:00:00Z",
        "ended_at": None,
        "preferred_not": [],
        "failure": None,
    }
    if status == "failed":
        job["failure"] = {"kind": kind or "hugin_crash", "message": "failed", "details": {}}
    return job


def _payload(jobs):
    return {"schema_version": "1.3", "dispatch_id": "20260430-110509", "jobs": jobs}


def _ids_of_type(component, type_name: str) -> list[dict]:
    return [
        getattr(c, "id", None)
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict) and getattr(c, "id", {}).get("type") == type_name
    ]


def _text_blob(component) -> str:
    parts = []
    for c in _walk(component):
        children = getattr(c, "children", None)
        if isinstance(children, str):
            parts.append(children)
    return " ".join(parts)


def test_running_row_has_tail_button():
    component = render_jobs_section(_payload([_job(run_id="rid-running", status="running")]))

    assert _ids_of_type(component, "tab-a-running-tail-toggle") == [
        {"type": "tab-a-running-tail-toggle", "run_id": "rid-running"}
    ]


def test_completed_row_does_not_have_tail_button():
    component = render_jobs_section(_payload([_job(run_id="rid-done", status="completed")]))

    assert _ids_of_type(component, "tab-a-running-tail-toggle") == []


def test_failed_row_has_retry_and_expand_but_not_tail_button():
    component = render_jobs_section(_payload([_job(run_id="rid-failed", status="failed", kind="timeout")]))

    assert _ids_of_type(component, "tab-a-expand-toggle") == [
        {"type": "tab-a-expand-toggle", "run_id": "rid-failed"}
    ]
    assert _ids_of_type(component, "tab-a-retry-toggle") == [
        {"type": "tab-a-retry-toggle", "run_id": "rid-failed"}
    ]
    assert _ids_of_type(component, "tab-a-running-tail-toggle") == []


def test_expand_running_row_renders_lines_in_pre_block():
    component = _expand_running_row(
        _job(run_id="rid-running"),
        {"source": "training.stdout.log", "lines": ["iter 1", "iter 2"], "fetched_at": "2026-04-30T12:01:00Z"},
    )

    pre_blocks = [c for c in _walk(component) if type(c).__name__ == "Pre"]
    assert len(pre_blocks) == 1
    assert pre_blocks[0].children == "iter 1\niter 2"
    assert _ids_of_type(component, "tab-a-running-tail-refresh") == [
        {"type": "tab-a-running-tail-refresh", "run_id": "rid-running"}
    ]


def test_expand_running_row_shows_filename_marker():
    for source in ("training.stdout.log", "startup.stdout.log"):
        component = _expand_running_row(
            _job(run_id=f"rid-{source}"),
            {"source": source, "lines": ["x"], "fetched_at": "2026-04-30T12:01:00Z"},
        )
        assert source in _text_blob(component)


def test_render_jobs_section_inserts_running_expand_row_when_shown():
    component = render_jobs_section(
        _payload([_job(run_id="rid-open", status="running")]),
        running_tail_shown={"rid-open"},
        running_tail_store={
            "rid-open": {
                "source": "training.stdout.log",
                "lines": ["reward=1.0"],
                "fetched_at": "2026-04-30T12:01:00Z",
            }
        },
    )

    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    assert len(rows) == 3
    assert "reward=1.0" in _text_blob(component)
```

- [ ] **Step 2: Add failing layout-store and callback tests**

In `tools/odin/valhalla/dashboard/tests/test_app.py`, extend `test_tab_a_render_returns_layout_with_expected_slots()`:

```python
    assert _has_id(component, "tab-a-running-tail-shown")
    assert _has_id(component, "tab-a-running-tail-store")
```

In `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`, update `_StubData`:

```python
    def __init__(self, dispatch_payload, *, hardware=None, lookup_results=None):
        self._dp = dispatch_payload
        self._hw = hardware
        self._lookup = lookup_results or {}
        self.load_dispatch_calls: list[str] = []
        self.load_hardware_calls: list[str] = []
        self.lookup_hardware_calls: list[str] = []
        self.running_tail_calls: list[tuple] = []
        self.running_tail_payload = {"source": "training.stdout.log", "lines": ["iter 1"]}
        self._runs_root = Path("/tmp")

    def read_running_job_tail_payload(self, dispatch_id: str, run_id: str, **kwargs) -> dict:
        self.running_tail_calls.append((dispatch_id, run_id, kwargs))
        return dict(self.running_tail_payload)
```

Add these tests to the same file:

```python
def test_running_tail_toggle_adds_then_removes():
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    ids = [{"type": "tab-a-running-tail-toggle", "run_id": "run-1"}]
    assert cb_mod._on_running_tail_toggle_handler([1], ids, current=[]) == ["run-1"]
    assert cb_mod._on_running_tail_toggle_handler([1], ids, current=["run-1"]) == []


def test_running_tail_callback_round_trip():
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    data = _StubData(_payload([_job(run_id="run-1", status="running")]))
    ids = [{"type": "tab-a-running-tail-toggle", "run_id": "run-1"}]

    shown = cb_mod._on_running_tail_toggle_handler([1], ids, current=[])
    store = cb_mod._on_running_tail_fetch_handler(
        [1],
        [],
        ids,
        [],
        dispatch_id="d",
        current_shown=[],
        current_store={},
        data=data,
        triggered_id=ids[0],
    )

    assert shown == ["run-1"]
    assert store["run-1"]["source"] == "training.stdout.log"
    assert store["run-1"]["lines"] == ["iter 1"]
    assert "fetched_at" in store["run-1"]
    assert data.running_tail_calls == [
        ("d", "run-1", {"host": "v1", "ssh_user": "horde", "container_name": "isaac-lab-base", "n": 50})
    ]

    hidden = cb_mod._on_running_tail_toggle_handler([1], ids, current=shown)
    hidden_fetch = cb_mod._on_running_tail_fetch_handler(
        [1],
        [],
        ids,
        [],
        dispatch_id="d",
        current_shown=shown,
        current_store=store,
        data=data,
        triggered_id=ids[0],
    )

    import dash

    assert hidden == []
    assert hidden_fetch is dash.no_update


def test_running_tail_refresh_refetches_existing_store():
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    data = _StubData(_payload([_job(run_id="run-2", status="running")]))
    data.running_tail_payload = {"source": "startup.stdout.log", "lines": ["booting"]}
    refresh_id = {"type": "tab-a-running-tail-refresh", "run_id": "run-2"}

    store = cb_mod._on_running_tail_fetch_handler(
        [],
        [1],
        [],
        [refresh_id],
        dispatch_id="d",
        current_shown=["run-2"],
        current_store={"run-2": {"source": "training.stdout.log", "lines": ["old"], "fetched_at": "old"}},
        data=data,
        triggered_id=refresh_id,
    )

    assert store["run-2"]["source"] == "startup.stdout.log"
    assert store["run-2"]["lines"] == ["booting"]


def test_running_tail_fetch_ignores_phantom_click():
    import dash

    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    data = _StubData(_payload([_job(run_id="run-3", status="running")]))
    out = cb_mod._on_running_tail_fetch_handler(
        [],
        [],
        [],
        [],
        dispatch_id="d",
        current_shown=[],
        current_store={},
        data=data,
        triggered_id=None,
    )

    assert out is dash.no_update
    assert data.running_tail_calls == []
```

- [ ] **Step 3: Run each new Tab A test and verify it fails**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_running_row_has_tail_button -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_completed_row_does_not_have_tail_button -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_failed_row_has_retry_and_expand_but_not_tail_button -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_expand_running_row_renders_lines_in_pre_block -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_expand_running_row_shows_filename_marker -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_render_jobs_section_inserts_running_expand_row_when_shown -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_app.py::test_tab_a_render_returns_layout_with_expected_slots -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_toggle_adds_then_removes -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_callback_round_trip -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_refresh_refetches_existing_store -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_fetch_ignores_phantom_click -v --tb=short --noconftest -p no:cacheprovider
```

Expected: fail because layout stores, running-tail render helpers, and callback helpers do not exist yet.

- [ ] **Step 4: Add Tab A stores**

In `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`, add the two stores after `tab-a-ssh-tail-store`:

```python
            dcc.Store(id="tab-a-running-tail-shown", storage_type="memory", data=[]),
            dcc.Store(id="tab-a-running-tail-store", storage_type="memory", data={}),
```

- [ ] **Step 5: Add running-row button and expand row rendering**

In `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`, extend both `render_jobs_section()` and `render_jobs_rows()` signatures:

```python
    running_tail_shown: set[str] | None = None,
    running_tail_store: dict[str, dict] | None = None,
```

Initialize the new values beside the existing stores:

```python
    running_tail_shown = running_tail_shown or set()
    running_tail_store = running_tail_store or {}
```

In both row-building loops, append a running expand row after the failed-row case:

```python
        if j.get("status") == "failed" and j.get("run_id") in expanded_run_ids:
            body_rows.append(_expand_row(j, ssh_tail_store.get(j.get("run_id"))))
        if j.get("status") == "running" and j.get("run_id") in running_tail_shown:
            body_rows.append(_expand_running_row(j, running_tail_store.get(j.get("run_id"))))
```

In `_data_row()`, replace the `else: failure_cell = "—"` branch with this running-row branch:

```python
    elif status == "running":
        failure_cell = html.Button(
            "👁",
            id={"type": "tab-a-running-tail-toggle", "run_id": run_id},
            n_clicks=0,
            className="tab-a-expand-toggle tab-a-running-tail-toggle",
            title="Show / hide running stdout tail",
        )
    else:
        failure_cell = "—"
```

Add `_expand_running_row()` below `_expand_row()`:

```python
def _expand_running_row(job: dict, tail_entry: dict | None) -> html.Tr:
    """Inline expansion row for a running job's current Hugin stdout tail."""
    run_id = job.get("run_id", "")
    entry = tail_entry or {}
    source = entry.get("source") or "training.stdout.log"
    fetched_at = entry.get("fetched_at")
    lines = entry.get("lines") if "lines" in entry else None

    body: list = [
        html.Span("Live tail ", className="tab-a-expand-label"),
        html.Span(str(source), className="tab-a-running-tail-source"),
    ]
    if fetched_at:
        body.extend(
            [
                html.Span("  Fetched ", className="tab-a-expand-label"),
                html.Span(str(fetched_at), className="tab-a-muted"),
            ]
        )
    body.extend(
        [
            html.Br(),
            html.Button(
                "Refresh",
                id={"type": "tab-a-running-tail-refresh", "run_id": run_id},
                n_clicks=0,
                className="tab-a-ssh-tail-button tab-a-running-tail-refresh",
            ),
        ]
    )

    if lines is None:
        body.append(html.P("Tail not loaded yet.", className="tab-a-ssh-tail-empty"))
    elif lines:
        body.append(html.Pre("\n".join(lines), className="tab-a-ssh-tail-pre tab-a-running-tail-pre"))
    else:
        body.append(
            html.P(
                "No running stdout log is available yet. The job may not have reached Hugin's phase runner.",
                className="tab-a-ssh-tail-empty",
            )
        )

    return html.Tr(
        className="tab-a-expand-row tab-a-running-tail-row",
        children=[html.Td(colSpan=7, children=body)],
    )
```

- [ ] **Step 6: Wire callback state and pure helpers**

In `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`, add imports:

```python
from datetime import datetime, timezone
```

Add `Input("tab-a-running-tail-shown", "data")` and `Input("tab-a-running-tail-store", "data")` to `_update_jobs`, then add parameters `running_tail_shown` and `running_tail_store`. Pass them into `_compute_jobs_children()`.

Update `_compute_jobs_children()` signature:

```python
    running_tail_shown: list[str] | None = None,
    running_tail_store: dict[str, dict] | None = None,
```

Pass them into `render_jobs_rows()`:

```python
        running_tail_shown=set(running_tail_shown or []),
        running_tail_store=running_tail_store or {},
```

Add the two callback registrations inside `register_callbacks()`:

```python
    @app.callback(
        Output("tab-a-running-tail-shown", "data"),
        Input({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "id"),
        State("tab-a-running-tail-shown", "data"),
    )
    def _on_running_tail_toggle(n_clicks_list, ids_list, current):
        return _on_running_tail_toggle_handler(n_clicks_list, ids_list, current=current)

    @app.callback(
        Output("tab-a-running-tail-store", "data"),
        Input({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "n_clicks"),
        Input({"type": "tab-a-running-tail-refresh", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "id"),
        State({"type": "tab-a-running-tail-refresh", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-running-tail-shown", "data"),
        State("tab-a-running-tail-store", "data"),
        prevent_initial_call=True,
    )
    def _on_running_tail_fetch(
        toggle_clicks,
        refresh_clicks,
        toggle_ids,
        refresh_ids,
        dispatch_id,
        current_shown,
        current_store,
    ):
        return _on_running_tail_fetch_handler(
            toggle_clicks,
            refresh_clicks,
            toggle_ids,
            refresh_ids,
            dispatch_id=dispatch_id,
            current_shown=current_shown,
            current_store=current_store,
            data=data,
            triggered_id=dash.ctx.triggered_id,
        )
```

Add these pure helpers near the existing `_on_ssh_tail_handler()` helpers:

```python
def _on_running_tail_toggle_handler(n_clicks_list, ids_list, *, current):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            return _toggle_run_id(current or [], ident["run_id"])
    return dash.no_update


def _on_running_tail_fetch_handler(
    toggle_clicks,
    refresh_clicks,
    toggle_ids,
    refresh_ids,
    *,
    dispatch_id,
    current_shown,
    current_store,
    data,
    triggered_id,
):
    ident = triggered_id or _last_clicked_id(refresh_clicks, refresh_ids) or _last_clicked_id(toggle_clicks, toggle_ids)
    if not isinstance(ident, dict) or not dispatch_id:
        return dash.no_update
    run_id = ident.get("run_id")
    if not run_id:
        return dash.no_update
    if ident.get("type") == "tab-a-running-tail-toggle" and run_id in set(current_shown or []):
        return dash.no_update
    return _compute_running_tail_store(data, dispatch_id, run_id, current_store=current_store or {})


def _last_clicked_id(n_clicks_list, ids_list):
    if not n_clicks_list or not any(n_clicks_list):
        return None
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            return ident
    return None


def _compute_running_tail_store(data, dispatch_id: str, run_id: str, *, current_store: dict):
    payload = data.load_dispatch(dispatch_id)
    job = _find_job(payload, run_id)
    new_store = dict(current_store or {})
    if job is None or not job.get("assigned_to"):
        new_store[run_id] = {
            "source": None,
            "lines": [],
            "fetched_at": _utc_now_iso(),
        }
        return new_store
    entry = data.read_running_job_tail_payload(
        dispatch_id,
        run_id,
        host=str(job.get("assigned_to")),
        ssh_user="horde",
        container_name="isaac-lab-base",
        n=50,
    )
    new_store[run_id] = {
        "source": entry.get("source"),
        "lines": list(entry.get("lines") or []),
        "fetched_at": _utc_now_iso(),
    }
    return new_store


def _find_job(payload: dict, run_id: str) -> dict | None:
    for job in payload.get("jobs", []) or []:
        if job.get("run_id") == run_id:
            return job
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
```

- [ ] **Step 7: Add CSS aliases**

In `tools/odin/valhalla/dashboard/assets/style.css`, add these rules near the existing ssh-tail styles:

```css
.tab-a-running-tail-toggle {
  min-width: 26px;
  cursor: pointer;
}
.tab-a-running-tail-source {
  color: #66b6ff;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}
.tab-a-running-tail-refresh {
  margin-right: 8px;
}
```

- [ ] **Step 8: Run each Tab A test and verify it passes**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_running_row_has_tail_button -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_completed_row_does_not_have_tail_button -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_failed_row_has_retry_and_expand_but_not_tail_button -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_expand_running_row_renders_lines_in_pre_block -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_expand_running_row_shows_filename_marker -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py::test_render_jobs_section_inserts_running_expand_row_when_shown -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_app.py::test_tab_a_render_returns_layout_with_expected_slots -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_toggle_adds_then_removes -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_callback_round_trip -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_refresh_refetches_existing_store -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_running_tail_fetch_ignores_phantom_click -v --tb=short --noconftest -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 9: Run nearby existing Tab A regression tests**

Run these one at a time:

```bash
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py::test_retry_button_only_on_failed_rows -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py::test_jobs_expanded_row_ssh_tail_button_present -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_retry_toggle_round_trip -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py::test_load_ssh_tail_callback_writes_store -v --tb=short --noconftest -p no:cacheprovider
```

Expected: all pass. These protect retry toggle and failed-row ssh-tail behavior from the new running-row UI.

- [ ] **Step 10: Run pre-commit for Tab A files**

```bash
pre-commit run --files tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py tools/odin/valhalla/dashboard/assets/style.css tools/odin/valhalla/dashboard/tests/test_app.py tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py
```

Expected: pass. If formatting changes touched files, review the diff and rerun the command.

- [ ] **Step 11: Run two review gates for Task 3**

Spec-compliance review prompt:

```text
Review Task 3 only. Compare the Tab A diff against Part C of docs/superpowers/specs/2026-04-30-odin-live-job-tail-design.md and Task 3 in the implementation plan. Verify running-only button behavior, failed-row ssh-tail preservation, retry preservation, expand-row render behavior, refresh behavior, and store state shape.
```

Code-quality review prompt:

```text
Review Task 3 only for code quality. Focus on Dash callback trigger semantics, pure helper testability, unnecessary SSH fetches on collapse, rendering robustness when host or lines are missing, and CSS consistency with the existing Tab A table.
```

Address review findings before committing.

- [ ] **Step 12: Commit Task 3**

```bash
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py tools/odin/valhalla/dashboard/assets/style.css tools/odin/valhalla/dashboard/tests/test_app.py tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py
git commit -m "Show running job tails in Tab A" \
  -m "Add a running-row tail toggle, refreshable expand row, and state." \
  -m "Read bundle-side Hugin stdout without changing failed-row ssh-tail."
```

---

## Final Verification

- [ ] **Step 1: Run the full touched-test list one at a time**

Run every test ID listed in Tasks 1, 2, and 3. Expected: all pass.

- [ ] **Step 2: Run final pre-commit for all touched files**

```bash
pre-commit run --files tools/odin/hugin/run.py tools/odin/tests/test_hugin.py tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data_running_tail.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py tools/odin/valhalla/dashboard/assets/style.css tools/odin/valhalla/dashboard/tests/test_app.py tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py
```

Expected: pass.

- [ ] **Step 3: Inspect final diff**

```bash
git diff --stat HEAD~3..HEAD
git diff HEAD~3..HEAD -- tools/odin/hugin/run.py tools/odin/tests/test_hugin.py tools/odin/valhalla/dashboard/data.py tools/odin/valhalla/dashboard/tests/test_data_running_tail.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py tools/odin/valhalla/dashboard/assets/style.css tools/odin/valhalla/dashboard/tests/test_app.py tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py
```

Expected: only the planned files changed. No unrelated generated reports, images, local run artifacts, or workspace files are staged.

- [ ] **Step 4: Record rollout caveat in final handoff**

State explicitly that the Hugin code runs inside the Valkyrie container image, so live hosts need a container rebuild before new dispatches produce streaming `startup.stdout.log` and `training.stdout.log` files. Unit tests cover the behavior without requiring a live host.
