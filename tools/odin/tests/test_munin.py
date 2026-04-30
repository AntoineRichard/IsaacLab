# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for Munin: mocks subprocess.run to avoid real trainings."""

import json
import os

import pytest

from tools.odin.munin import run as munin_run


class _Completed:
    """Fake :class:`subprocess.CompletedProcess` for the streaming-fh path."""

    def __init__(self, returncode: int):
        self.returncode = returncode
        # Streaming sends bytes through stdout/stderr file handles, so the
        # returned object's stdout/stderr are unused (but kept here as None
        # to match real CompletedProcess shape).
        self.stdout = None
        self.stderr = None


def _fake_run_factory():
    """Return a stub that pretends to write startup.json/training.json and
    records every command it was called with for later assertions."""

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        out_idx = cmd.index("--schema_v1_output") + 1
        out_path = cmd[out_idx]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write('{"schema_version": "1.0", "fake": true}\n')
        # Munin now streams to file handles; write happy-path bytes there
        # so the per-phase logs show up on disk like the real runner.
        if "stdout" in kwargs:
            kwargs["stdout"].write(b"fake stdout\n")
        if "stderr" in kwargs:
            kwargs["stderr"].write(b"fake stderr\n")
        return _Completed(0)

    _fake_run.captured_cmds = captured_cmds
    return _fake_run


def test_munin_happy_path(tmp_path, monkeypatch):
    bundle_root = str(tmp_path)
    fake_run = _fake_run_factory()
    monkeypatch.setattr(munin_run, "_subprocess_run", fake_run)
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
    munin_run.main()

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("skrl_physx_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    assert os.path.exists(os.path.join(bundle, "startup.json"))
    assert os.path.exists(os.path.join(bundle, "training.json"))
    assert os.path.exists(os.path.join(bundle, "manifest.json"))
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["exit_code"] == 0
    assert m["config"]["framework"] == "skrl"

    training_data_dir = os.path.join(bundle, "training_data")
    assert os.path.isdir(training_data_dir), f"{training_data_dir} not created"
    # Training subprocess should receive --log_dir <bundle>/training_data.
    training_cmds = [c for c in fake_run.captured_cmds if "benchmark_skrl.py" in " ".join(c)]
    assert len(training_cmds) == 1
    cmd = training_cmds[0]
    assert "--log_dir" in cmd
    log_dir_idx = cmd.index("--log_dir")
    assert cmd[log_dir_idx + 1] == training_data_dir
    # Old tb/ directory should no longer be created.
    assert not os.path.exists(os.path.join(bundle, "tb")), "stale tb/ dir leaked"
    # The manifest's artifacts list should reflect the new dir.
    assert "training_data" in m["artifacts"]


def test_run_phase_streams_stdout_to_log_during_run(tmp_path, monkeypatch):
    """Bytes written to ``stdout=`` must be visible on disk WHILE the
    subprocess is running, not only after it exits. The dashboard's live-tail
    reader expects this — Munin previously buffered everything via
    ``capture_output=True`` and only wrote a tail on failure, so a running
    skrl job's training.stdout.log was 0 bytes for the whole run."""
    output_json = os.path.join(tmp_path, "training.json")

    def _streaming_run(cmd, *args, **kwargs):
        stdout_fh = kwargs["stdout"]
        stdout_fh.write(b"first line\n")
        stdout_fh.flush()
        with open(stdout_fh.name, "rb") as f:
            assert f.read() == b"first line\n"
        # Materialize the schema_v1_output the phase expects.
        with open(output_json, "w") as f:
            f.write('{"schema_version": "1.0"}')
        return _Completed(0)

    monkeypatch.setattr(munin_run, "_subprocess_run", _streaming_run)

    phase = munin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )
    assert phase.status == "completed"


def test_run_phase_writes_full_stdout_on_completed(tmp_path, monkeypatch):
    """Happy-path stdout must land on disk in full (not tail-truncated).
    Older Munin only wrote logs on failure; once the wipe-on-submit fix
    started flagging mid-run hugin_crash on stale partial logs, having
    full happy-path logs available is essential for live tail to work."""
    output_json = os.path.join(tmp_path, "training.json")

    def _run(cmd, *args, **kwargs):
        kwargs["stdout"].write(b"line 1\nline 2\n")
        kwargs["stderr"].write(b"warning\n")
        with open(output_json, "w") as f:
            f.write('{"schema_version": "1.0"}')
        return _Completed(0)

    monkeypatch.setattr(munin_run, "_subprocess_run", _run)

    munin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    with open(os.path.join(tmp_path, "logs", "training.stdout.log"), "rb") as f:
        assert f.read() == b"line 1\nline 2\n"
    with open(os.path.join(tmp_path, "logs", "training.stderr.log"), "rb") as f:
        assert f.read() == b"warning\n"


def test_run_phase_injects_dash_u_for_python_child(tmp_path, monkeypatch):
    """File-redirected Python block-buffers stdout by default; ``-u`` is
    necessary to keep iteration-level lines visible in the live-tail."""
    output_json = os.path.join(tmp_path, "training.json")
    captured_cmds: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        kwargs["stdout"].write(b"x")
        kwargs["stderr"].write(b"y")
        with open(output_json, "w") as f:
            f.write('{"schema_version": "1.0"}')
        return _Completed(0)

    monkeypatch.setattr(munin_run, "_subprocess_run", _run)
    munin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )
    assert captured_cmds[0][:2] == ["python.sh", "-u"]


def test_munin_failure_path_writes_logs(tmp_path, monkeypatch):
    def _failing_run(cmd, *args, **kwargs):
        if "stdout" in kwargs:
            kwargs["stdout"].write(b"partial stdout")
        if "stderr" in kwargs:
            kwargs["stderr"].write(b"traceback...")
        return _Completed(7)

    bundle_root = str(tmp_path)
    monkeypatch.setattr(munin_run, "_subprocess_run", _failing_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "munin",
            "--task",
            "Isaac-Ant-Direct-v0",
            "--backend",
            "newton",
            "--seed",
            "1",
            "--runs_root",
            bundle_root,
            "--skip_startup",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        munin_run.main()
    assert exc.value.code == 7

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("skrl_newton_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    assert os.path.exists(os.path.join(bundle, "logs", "training.stderr.log"))
    with open(os.path.join(bundle, "logs", "training.stderr.log"), "rb") as f:
        assert f.read() == b"traceback..."
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["status"] == "failed"
    assert m["phases"]["training"]["exit_code"] == 7


def test_munin_honors_run_id_override(tmp_path, monkeypatch):
    """--run_id uses the string verbatim instead of compute_run_id."""
    bundle_root = str(tmp_path)
    fake_run = _fake_run_factory()
    monkeypatch.setattr(munin_run, "_subprocess_run", fake_run)
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
            "--runs_root",
            bundle_root,
            "--skip_startup",
            "--run_id",
            "dispatched-run-id-xyz",
        ],
    )
    munin_run.main()

    assert os.path.isdir(os.path.join(bundle_root, "dispatched-run-id-xyz"))
    siblings = [d for d in os.listdir(bundle_root) if d != "dispatched-run-id-xyz"]
    assert siblings == [], f"unexpected sibling bundle dirs: {siblings}"


def test_munin_silent_exit_zero_no_output_marks_failed(tmp_path, monkeypatch):
    """Subprocess exits 0 but writes no output JSON → phase status='failed'."""

    def _silent_exit_zero(cmd, *args, **kwargs):
        # Do NOT create the --schema_v1_output file.
        return _Completed(0)

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
