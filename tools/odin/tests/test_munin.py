# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for Munin: mocks subprocess.run to avoid real trainings."""

import json
import os

import pytest

from tools.odin.munin import run as munin_run


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

        class R:
            returncode = 0
            stdout = b"fake stdout"
            stderr = b"fake stderr"

        return R()

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


def test_munin_failure_path_writes_logs(tmp_path, monkeypatch):
    def _failing_run(cmd, *args, **kwargs):
        class R:
            returncode = 7
            stdout = b"partial stdout"
            stderr = b"traceback..."

        return R()

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
