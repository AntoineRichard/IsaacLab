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
    """Return a stub that pretends to write startup.json/training.json."""

    def _fake_run(cmd, *args, **kwargs):
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

    return _fake_run


def test_hugin_happy_path(tmp_path, monkeypatch):
    bundle_root = str(tmp_path)
    monkeypatch.setattr(hugin_run, "_subprocess_run", _fake_run_factory())
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
