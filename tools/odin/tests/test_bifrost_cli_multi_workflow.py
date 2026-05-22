# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI orchestration: one OSMO workflow per chunk; exec_timeout = chunk max."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.odin.asgard.state import read_dispatch_state
from tools.odin.bifrost import cli as bifrost_cli
from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot

_CFG = """\
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL
image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
defaults:
  resources:
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000
  exec_timeout: 14400
  queue_timeout: 7200
retry: {reschedule_codes: "3001-3006", restart_codes: ""}
bundle_dataset_prefix: odin
code_delivery: {mode: files_upload, source_root: tools/odin}
chunk_size: 1
"""

_BUDGETS = """\
defaults:
  rsl_rl: 43200
budgets:
  Isaac-Cartpole-Direct-v0:
    rsl_rl: 1800   # 30m → smaller bucket
  Isaac-Ant-Direct-v0:
    rsl_rl: 7200   # 2h → larger bucket
"""


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(_CFG)
    return p


@pytest.fixture
def budgets_path(tmp_path: Path) -> Path:
    p = tmp_path / "job_budgets.yaml"
    p.write_text(_BUDGETS)
    return p


@pytest.fixture
def physx_yaml(tmp_path: Path) -> Path:
    """Two kept envs → chunk_size=1 → two chunks → two workflows."""
    p = tmp_path / "physx.yaml"
    p.write_text(
        "groups:\n"
        "  cartpole:\n"
        "  - task_id: Isaac-Cartpole-Direct-v0\n"
        "    framework: rsl_rl\n"
        "    num_envs: 4096\n"
        "    max_iterations: 150\n"
        "    keep: true\n"
        "  ant:\n"
        "  - task_id: Isaac-Ant-Direct-v0\n"
        "    framework: rsl_rl\n"
        "    num_envs: 4096\n"
        "    max_iterations: 1000\n"
        "    keep: true\n"
    )
    return p


class _RecordingClient:
    """In-memory ``OsmoClient`` stand-in.

    Records every submit + status call, returns terminal status on the
    second poll so the loop exits quickly.
    """

    def __init__(self, *_a, **_k):
        self.submit_calls: list[Path] = []
        self.submitted_yaml: list[str] = []
        self.status_calls: list[str] = []
        self._submit_counter = 0

    def submit(self, yaml_path, *, rsync_pairs=(), pool=None):
        self.submit_calls.append(yaml_path)
        self.submitted_yaml.append(Path(yaml_path).read_text())
        self._submit_counter += 1
        return f"wf-mock-{self._submit_counter}"

    def status(self, workflow_id):
        self.status_calls.append(workflow_id)
        return WorkflowSnapshot(workflow_id, "COMPLETED", _all_tasks_completed(workflow_id))

    def dataset_download(self, name, dest_dir):
        # Find the run_id from the dataset name suffix and stamp a manifest.
        # Dataset name is f"{prefix}-{dispatch_id}-{run_id}".
        run_id = name.split("-", 2)[-1].split("-", 1)[-1]
        for guess in (name.split("-")[-1], run_id):
            d = Path(dest_dir) / guess
            try:
                d.mkdir(parents=True, exist_ok=True)
                (d / "manifest.json").write_text("{}")
            except OSError:
                pass


_TASK_NAMES_BY_WF: dict[str, list[str]] = {}


def _all_tasks_completed(workflow_id: str) -> list[TaskSnapshot]:
    return [TaskSnapshot(name=n, status="COMPLETED", exit_code=0) for n in _TASK_NAMES_BY_WF.get(workflow_id, [])]


def _wire_fake_client(monkeypatch_holder: list[_RecordingClient]):
    """Patch ``OsmoClient`` and capture every submitted YAML's task names."""
    import yaml as _yaml

    _TASK_NAMES_BY_WF.clear()

    def factory(*a, **k):
        c = _RecordingClient(*a, **k)
        monkeypatch_holder.append(c)
        orig_submit = c.submit

        def _submit(yaml_path, *, rsync_pairs=(), pool=None):
            wf_id = orig_submit(yaml_path, rsync_pairs=rsync_pairs, pool=pool)
            parsed = _yaml.safe_load(Path(yaml_path).read_text())
            # Each task is wrapped in its own one-task group; flatten the names.
            _TASK_NAMES_BY_WF[wf_id] = [
                t["name"] for g in parsed["workflow"]["groups"] for t in g["tasks"]
            ]
            return wf_id

        c.submit = _submit
        return c

    return factory


def test_two_buckets_submit_two_workflows_with_shared_exec_timeout(
    tmp_path: Path,
    cfg_path: Path,
    physx_yaml: Path,
    budgets_path: Path,
):
    """chunk_size=1 + two envs → two workflows that share the global exec_timeout.

    Each task is wrapped in its own one-task group, so the OSMO
    ``exec_timeout`` clock is per-group; the workflow value is a fixed
    global ceiling sourced from ``cfg.defaults.exec_timeout`` and short
    tasks finish well within it.
    """
    import yaml as _yaml

    runs_root = tmp_path / "odin_runs"
    clients: list[_RecordingClient] = []
    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=_wire_fake_client(clients)):
        rc = bifrost_cli.main(
            [
                "--osmo-config",
                str(cfg_path),
                "--physx-yaml",
                str(physx_yaml),
                "--budgets-yaml",
                str(budgets_path),
                "--seeds",
                "42",
                "--runs-root",
                str(runs_root),
                "--poll-interval",
                "0",
            ]
        )
    assert rc == 0
    [client] = clients
    assert len(client.submit_calls) == 2, "expected one submit per chunk"
    timeouts = []
    for body in client.submitted_yaml:
        parsed = _yaml.safe_load(body)
        timeouts.append(parsed["workflow"]["timeout"]["exec_timeout"])
    # Fixture cfg sets defaults.exec_timeout = 14400s. Both chunks
    # use that fixed ceiling regardless of per-row budget differences.
    assert timeouts == ["14400s", "14400s"]


def test_state_osmo_workflow_ids_populated(
    tmp_path: Path,
    cfg_path: Path,
    physx_yaml: Path,
    budgets_path: Path,
):
    runs_root = tmp_path / "odin_runs"
    clients: list[_RecordingClient] = []
    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=_wire_fake_client(clients)):
        bifrost_cli.main(
            [
                "--osmo-config",
                str(cfg_path),
                "--physx-yaml",
                str(physx_yaml),
                "--budgets-yaml",
                str(budgets_path),
                "--seeds",
                "42",
                "--runs-root",
                str(runs_root),
                "--poll-interval",
                "0",
            ]
        )
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    assert state is not None
    assert state.osmo_workflow_ids == ["wf-mock-1", "wf-mock-2"]


def test_resume_walks_multiple_workflow_ids(
    tmp_path: Path,
    cfg_path: Path,
    physx_yaml: Path,
    budgets_path: Path,
):
    """``--resume LATEST`` re-attaches the poller to every wf id on disk."""
    from tools.odin.asgard.state import write_dispatch_state

    runs_root = tmp_path / "odin_runs"
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(cfg_path),
            "--physx-yaml",
            str(physx_yaml),
            "--budgets-yaml",
            str(budgets_path),
            "--seeds",
            "42",
            "--runs-root",
            str(runs_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    assert state is not None
    state.osmo_workflow_ids = ["wf-resume-a", "wf-resume-b"]
    write_dispatch_state(dispatch_dir, state)

    # The two rows: Cartpole (chunk 0) and Ant (chunk 1) — split across
    # workflows so each wf's status() reports its own task as COMPLETED.
    by_task: dict[str, str] = {j.task_id: j.osmo_task_name for j in state.jobs}
    _TASK_NAMES_BY_WF.clear()
    _TASK_NAMES_BY_WF["wf-resume-a"] = [by_task["Isaac-Cartpole-Direct-v0"]]
    _TASK_NAMES_BY_WF["wf-resume-b"] = [by_task["Isaac-Ant-Direct-v0"]]

    class _ResumeFake:
        def __init__(self, *_a, **_k):
            self.status_calls: list[str] = []
            self.submit_calls = 0

        def submit(self, *a, **k):
            self.submit_calls += 1
            raise AssertionError("resume must NOT call submit")

        def status(self, wf_id):
            self.status_calls.append(wf_id)
            return WorkflowSnapshot(wf_id, "COMPLETED", _all_tasks_completed(wf_id))

        def dataset_download(self, name, dest):
            run = name.split("-", 2)[-1].split("-", 1)[-1]
            (Path(dest) / run).mkdir(parents=True, exist_ok=True)
            (Path(dest) / run / "manifest.json").write_text("{}")

    holder: list[_ResumeFake] = []

    def _factory(*a, **k):
        c = _ResumeFake(*a, **k)
        holder.append(c)
        return c

    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=_factory):
        rc = bifrost_cli.main(
            [
                "--osmo-config",
                str(cfg_path),
                "--physx-yaml",
                str(physx_yaml),
                "--budgets-yaml",
                str(budgets_path),
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
    assert holder[0].submit_calls == 0
    assert set(holder[0].status_calls) == {"wf-resume-a", "wf-resume-b"}


def test_submit_failure_mid_dispatch_persists_earlier_workflow_id(
    tmp_path: Path,
    cfg_path: Path,
    physx_yaml: Path,
    budgets_path: Path,
):
    """If the second ``client.submit()`` raises, the first workflow id is on disk."""
    runs_root = tmp_path / "odin_runs"

    class _BlowUpOnSecond(_RecordingClient):
        def submit(self, yaml_path, *, rsync_pairs=(), pool=None):
            self.submit_calls.append(yaml_path)
            self.submitted_yaml.append(Path(yaml_path).read_text())
            self._submit_counter += 1
            if self._submit_counter == 1:
                return "wf-first-ok"
            raise RuntimeError("simulated second-submit failure")

    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=_BlowUpOnSecond):
        with pytest.raises(RuntimeError, match="second-submit"):
            bifrost_cli.main(
                [
                    "--osmo-config",
                    str(cfg_path),
                    "--physx-yaml",
                    str(physx_yaml),
                    "--budgets-yaml",
                    str(budgets_path),
                    "--seeds",
                    "42",
                    "--runs-root",
                    str(runs_root),
                    "--poll-interval",
                    "0",
                ]
            )
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    assert state is not None
    assert "wf-first-ok" in state.osmo_workflow_ids
