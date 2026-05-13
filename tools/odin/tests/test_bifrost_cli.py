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
        "groups:\n"
        "  direct/ant:\n"
        "  - task_id: Isaac-Ant-Direct-v0\n"
        "    framework: rsl-rl\n"
        "    num_envs: 4096\n"
        "    max_iterations: 500\n"
        "    keep: true\n"
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
    # One chunk per dispatch (chunk_size default 25 with one row → one workflow file).
    assert (dispatch_dirs[0] / "workflow.0.yaml").exists()


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
    workflow_yaml = list(runs_root.iterdir())[0] / "workflow.0.yaml"
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

        def submit(self, yaml_path, *, rsync_pairs=(), pool=None):
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


def test_resume_reattaches_to_existing_dispatch(tmp_path: Path, example_config: Path, example_physx_yaml: Path):
    """--resume LATEST should NOT create a new dispatch dir; it reuses the existing one."""
    from unittest.mock import patch

    from tools.odin.asgard.state import read_dispatch_state, write_dispatch_state
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
    assert state is not None
    state.osmo_workflow_id = "wf-already-running"
    write_dispatch_state(dispatch_dir, state)

    class FakeClient:
        def __init__(self, *_a, **_k):
            pass

        def submit(self, *a, **k):
            raise AssertionError("resume must NOT call submit")

        def status(self, wf):
            assert wf == "wf-already-running"
            cur = read_dispatch_state(dispatch_dir)
            assert cur is not None
            task_name = cur.jobs[0].osmo_task_name
            return WorkflowSnapshot(wf, "COMPLETED", [TaskSnapshot(task_name, "COMPLETED", 0)])

        def dataset_download(self, name, dest):
            cur = read_dispatch_state(dispatch_dir)
            assert cur is not None
            run = cur.jobs[0].run_id
            (Path(dest) / run).mkdir(parents=True, exist_ok=True)
            (Path(dest) / run / "manifest.json").write_text("{}")

    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=FakeClient):
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
    assert parent is not None
    # Use transition_to so the strict-invariants tripwire on
    # write_dispatch_state is satisfied (ended_at + failure required
    # for terminal states, set atomically by the helper).
    parent.jobs[0].transition_to("failed", failure=FailureInfo(kind="hugin_crash", message="boom", details={}))
    parent.jobs[1].transition_to("completed")
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
    assert child is not None
    assert child.parent_dispatch_id == parent.dispatch_id
    assert len(child.jobs) == 1
    assert child.jobs[0].run_id == failed_run_id


def test_verbose_tail_writes_log_file(tmp_path: Path, example_config: Path, example_physx_yaml: Path):
    from unittest.mock import patch

    from tools.odin.asgard.state import read_dispatch_state
    from tools.odin.bifrost import cli as bifrost_cli
    from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot

    runs_root = tmp_path / "odin_runs"

    class FakeClient:
        def __init__(self, *_a, **_k):
            self.calls = 0

        def submit(self, *a, **k):
            return "wf-1"

        def status(self, wf):
            self.calls += 1
            cur_dirs = sorted(runs_root.iterdir())
            cur = read_dispatch_state(cur_dirs[-1])
            assert cur is not None
            task_name = cur.jobs[0].osmo_task_name
            if self.calls == 1:
                # Allow tail thread to attach.
                return WorkflowSnapshot(wf, "RUNNING", [TaskSnapshot(task_name, "RUNNING", None)])
            return WorkflowSnapshot(wf, "COMPLETED", [TaskSnapshot(task_name, "COMPLETED", 0)])

        def logs(self, wf, task, *, follow):
            yield b"hello from osmo\n"

        def dataset_download(self, name, dest):
            cur = read_dispatch_state(Path(dest))
            assert cur is not None
            run = cur.jobs[0].run_id
            (Path(dest) / run).mkdir(parents=True, exist_ok=True)
            (Path(dest) / run / "manifest.json").write_text("{}")

    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=FakeClient):
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
    assert state is not None
    log_path = dispatch_dir / state.jobs[0].run_id / "logs" / "osmo-tail.log"
    # The tail thread is best-effort — give it a tiny window if needed.
    import time as _time

    deadline = _time.time() + 3.0
    while not log_path.exists() and _time.time() < deadline:
        _time.sleep(0.05)
    assert log_path.exists(), f"expected tail log at {log_path}"
    assert b"hello from osmo" in log_path.read_bytes()
