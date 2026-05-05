# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

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


def test_dry_run_writes_workflow_yaml_and_exits_zero(tmp_path: Path, example_config: Path, example_physx_yaml: Path):
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
    dispatch_dirs = list(runs_root.iterdir())
    assert len(dispatch_dirs) == 1
    assert (dispatch_dirs[0] / "workflow.yaml").exists()


def test_seed_expansion_creates_one_task_per_seed(tmp_path: Path, example_config: Path, example_physx_yaml: Path):
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


def test_main_submits_and_polls(tmp_path: Path, example_config: Path, example_physx_yaml: Path):
    from unittest.mock import patch

    from tools.odin.asgard.state import read_dispatch_state
    from tools.odin.bifrost import cli as bifrost_cli
    from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot

    runs_root = tmp_path / "odin_runs"

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.submit_calls: list[Path] = []
            self.status_calls = 0

        def submit(self, yaml_path, *, rsync_pairs=()):
            self.submit_calls.append(yaml_path)
            return "wf-test-1"

        def status(self, workflow_id):
            self.status_calls += 1
            state = read_dispatch_state(runs_root / sorted(p.name for p in runs_root.iterdir())[-1])
            assert state is not None
            task_name = state.jobs[0].osmo_task_name
            if self.status_calls == 1:
                return WorkflowSnapshot(
                    workflow_id,
                    "RUNNING",
                    [TaskSnapshot(task_name, "RUNNING", None)],
                )
            # Second (terminal) poll: task COMPLETED.
            return WorkflowSnapshot(
                workflow_id,
                "COMPLETED",
                [TaskSnapshot(task_name, "COMPLETED", 0)],
            )

        def dataset_download(self, name, dest_dir):
            # Find the run_id that the dispatcher allocated and write a fake manifest.
            state = read_dispatch_state(Path(dest_dir))
            if state and state.jobs:
                run_id = state.jobs[0].run_id
                run_dir = Path(dest_dir) / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "manifest.json").write_text("{}")

    fake_holder: list[FakeClient] = []

    def _factory(*a, **k):
        c = FakeClient(*a, **k)
        fake_holder.append(c)
        return c

    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=_factory):
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
    fake = fake_holder[0]
    assert fake.submit_calls
    assert fake.status_calls >= 2
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    assert state is not None
    assert state.osmo_workflow_id == "wf-test-1"
    assert state.jobs[0].status == "completed"
