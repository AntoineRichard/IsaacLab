# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for Hugin: mocks subprocess.run to avoid real trainings."""

import json
import os

import pytest

from tools.odin.hugin import run as hugin_run


class _Completed:
    def __init__(self, returncode: int):
        self.returncode = returncode


def _write_output_json_from_cmd(cmd):
    out_idx = cmd.index("--schema_v1_output") + 1
    out_path = cmd[out_idx]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write('{"schema_version": "1.0", "fake": true}\n')


def _fake_run_factory():
    """Return a stub that pretends to write startup.json/training.json and
    records every command it was called with for later assertions."""

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        _write_output_json_from_cmd(cmd)
        if "stdout" in kwargs:
            kwargs["stdout"].write(b"fake stdout")
        if "stderr" in kwargs:
            kwargs["stderr"].write(b"fake stderr")

        return _Completed(0)

    _fake_run.captured_cmds = captured_cmds
    return _fake_run


def test_run_phase_streams_stdout_to_log_during_run(tmp_path, monkeypatch):
    output_json = os.path.join(tmp_path, "training.json")

    def _streaming_run(cmd, *args, **kwargs):
        stdout_fh = kwargs["stdout"]
        stdout_fh.write(b"first line\n")
        stdout_fh.flush()
        with open(stdout_fh.name, "rb") as f:
            assert f.read() == b"first line\n"
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _streaming_run)

    phase = hugin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    assert phase.status == "completed"


def test_run_phase_writes_full_stdout_on_completed(tmp_path, monkeypatch):
    output_json = os.path.join(tmp_path, "training.json")

    def _run(cmd, *args, **kwargs):
        kwargs["stdout"].write(b"line 1\nline 2\n")
        kwargs["stderr"].write(b"warning\n")
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _run)

    hugin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    with open(os.path.join(tmp_path, "logs", "training.stdout.log"), "rb") as f:
        assert f.read() == b"line 1\nline 2\n"
    with open(os.path.join(tmp_path, "logs", "training.stderr.log"), "rb") as f:
        assert f.read() == b"warning\n"


def test_run_phase_no_tail_truncation_on_failure(tmp_path, monkeypatch):
    output_json = os.path.join(tmp_path, "training.json")
    stdout = b"".join(f"stdout {i}\n".encode() for i in range(2000))
    stderr = b"".join(f"stderr {i}\n".encode() for i in range(2000))

    def _run(cmd, *args, **kwargs):
        kwargs["stdout"].write(stdout)
        kwargs["stderr"].write(stderr)
        return _Completed(7)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _run)

    phase = hugin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    assert phase.status == "failed"
    assert phase.exit_code == 7
    with open(os.path.join(tmp_path, "logs", "training.stdout.log"), "rb") as f:
        assert f.read() == stdout
    with open(os.path.join(tmp_path, "logs", "training.stderr.log"), "rb") as f:
        assert f.read() == stderr


def test_run_phase_injects_dash_u_for_python_child(tmp_path, monkeypatch):
    output_json = os.path.join(tmp_path, "training.json")
    captured_cmds: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        kwargs["stdout"].write(b"stdout\n")
        kwargs["stderr"].write(b"stderr\n")
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _run)

    hugin_run._run_phase(
        cmd=["python.sh", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    assert captured_cmds == [["python.sh", "-u", "train.py", "--schema_v1_output", output_json]]


def test_run_phase_injects_dash_u_for_isaaclab_sh_python_mode(tmp_path, monkeypatch):
    output_json = os.path.join(tmp_path, "training.json")
    captured_cmds: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        kwargs["stdout"].write(b"stdout\n")
        kwargs["stderr"].write(b"stderr\n")
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _run)

    hugin_run._run_phase(
        cmd=["isaaclab.sh", "-p", "train.py", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    assert captured_cmds == [["isaaclab.sh", "-p", "-u", "train.py", "--schema_v1_output", output_json]]


def test_run_phase_does_not_inject_dash_u_for_non_python(tmp_path, monkeypatch):
    output_json = os.path.join(tmp_path, "training.json")
    captured_cmds: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        kwargs["stdout"].write(b"stdout\n")
        kwargs["stderr"].write(b"stderr\n")
        _write_output_json_from_cmd(cmd)
        return _Completed(0)

    monkeypatch.setattr(hugin_run, "_subprocess_run", _run)

    hugin_run._run_phase(
        cmd=["nvidia-smi", "--schema_v1_output", output_json],
        bundle_dir=str(tmp_path),
        phase_name="training",
        output_json=output_json,
    )

    assert captured_cmds == [["nvidia-smi", "--schema_v1_output", output_json]]


def test_hugin_happy_path(tmp_path, monkeypatch):
    bundle_root = str(tmp_path)
    fake_run = _fake_run_factory()
    monkeypatch.setattr(hugin_run, "_subprocess_run", fake_run)
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
    hugin_run.main()

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("rsl-rl_physx_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    assert os.path.exists(os.path.join(bundle, "startup.json"))
    assert os.path.exists(os.path.join(bundle, "training.json"))
    assert os.path.exists(os.path.join(bundle, "manifest.json"))
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["exit_code"] == 0

    training_data_dir = os.path.join(bundle, "training_data")
    assert os.path.isdir(training_data_dir), f"{training_data_dir} not created"
    # Training subprocess should receive --log_dir <bundle>/training_data.
    training_cmds = [c for c in fake_run.captured_cmds if "benchmark_rsl_rl.py" in " ".join(c)]
    assert len(training_cmds) == 1
    cmd = training_cmds[0]
    assert "--log_dir" in cmd
    log_dir_idx = cmd.index("--log_dir")
    assert cmd[log_dir_idx + 1] == training_data_dir
    # Old tb/ directory should no longer be created.
    assert not os.path.exists(os.path.join(bundle, "tb")), "stale tb/ dir leaked"
    # The manifest's artifacts list should reflect the new dir.
    assert "training_data" in m["artifacts"]


def test_hugin_failure_path_writes_logs(tmp_path, monkeypatch):
    def _failing_run(cmd, *args, **kwargs):
        kwargs["stdout"].write(b"partial stdout")
        kwargs["stderr"].write(b"traceback...")

        return _Completed(7)

    bundle_root = str(tmp_path)
    monkeypatch.setattr(hugin_run, "_subprocess_run", _failing_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hugin",
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
        hugin_run.main()
    assert exc.value.code == 7

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("rsl-rl_newton_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    assert os.path.exists(os.path.join(bundle, "logs", "training.stderr.log"))
    with open(os.path.join(bundle, "logs", "training.stderr.log"), "rb") as f:
        assert f.read() == b"traceback..."
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["status"] == "failed"
    assert m["phases"]["training"]["exit_code"] == 7


def test_hugin_honors_run_id_override(tmp_path, monkeypatch):
    """--run_id uses the string verbatim instead of compute_run_id."""
    bundle_root = str(tmp_path)
    fake_run = _fake_run_factory()
    monkeypatch.setattr(hugin_run, "_subprocess_run", fake_run)
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
            "--runs_root",
            bundle_root,
            "--skip_startup",
            "--run_id",
            "dispatched-run-id-xyz",
        ],
    )
    hugin_run.main()

    assert os.path.isdir(os.path.join(bundle_root, "dispatched-run-id-xyz"))
    # No auto-generated run_id sibling directory.
    siblings = [d for d in os.listdir(bundle_root) if d != "dispatched-run-id-xyz"]
    assert siblings == [], f"unexpected sibling bundle dirs: {siblings}"


def test_hugin_silent_exit_zero_no_output_marks_failed(tmp_path, monkeypatch):
    """Subprocess exits 0 but writes no output JSON → phase status='failed'."""

    def _silent_exit_zero(cmd, *args, **kwargs):
        # Do NOT create the --schema_v1_output file.
        return _Completed(0)

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
