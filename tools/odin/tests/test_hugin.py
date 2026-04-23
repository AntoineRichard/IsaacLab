# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for Hugin: mocks subprocess.run to avoid real trainings."""

import json
import os

import pytest

from tools.odin.hugin import run as hugin_run


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


def test_hugin_failure_path_writes_logs(tmp_path, monkeypatch):
    def _failing_run(cmd, *args, **kwargs):
        class R:
            returncode = 7
            stdout = b"partial stdout"
            stderr = b"traceback..."

        return R()

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
