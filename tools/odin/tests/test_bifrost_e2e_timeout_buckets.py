# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end timeout-bucket smoke against a mock OSMO client.

Walks the full Bifrost CLI flow — config load, planner (budgets lookup),
render, submit, poll, bundle download, aggregate — for a 6-task
dispatch split into two chunks. The mock client records every call so
we can assert:

- Two OSMO workflows submitted with the right ``exec_timeout`` values
  (chunk-max from ``job_budgets.yaml``).
- All six jobs land in ``completed`` state.
- ``aggregate.json`` is written with the expected totals.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml as _yaml

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
chunk_size: 3
"""

_ENVS = """\
groups:
  cartpole:
  - task_id: Isaac-Cartpole-Direct-v0
    framework: rsl_rl
    num_envs: 4096
    max_iterations: 150
    keep: true
  ant:
  - task_id: Isaac-Ant-Direct-v0
    framework: rsl_rl
    num_envs: 4096
    max_iterations: 1000
    keep: true
"""

_BUDGETS = """\
defaults:
  rsl_rl: 43200
budgets:
  Isaac-Cartpole-Direct-v0:
    rsl_rl: 1800   # 30m
  Isaac-Ant-Direct-v0:
    rsl_rl: 7200   # 2h
"""


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(_CFG)
    return p


@pytest.fixture
def envs_path(tmp_path: Path) -> Path:
    p = tmp_path / "physx.yaml"
    p.write_text(_ENVS)
    return p


@pytest.fixture
def budgets_path(tmp_path: Path) -> Path:
    p = tmp_path / "job_budgets.yaml"
    p.write_text(_BUDGETS)
    return p


class _FakeOsmoClient:
    """Records every submit/status/download call.

    Snoops each rendered workflow YAML on submit so ``status`` can echo
    back COMPLETED snapshots for whichever tasks live in the queried
    workflow (the multi-workflow loop calls ``status`` per workflow id).
    """

    def __init__(self, *_a, **_k):
        self.submits: list[tuple[Path, str]] = []  # (yaml_path, wf_id)
        self.status_calls: list[str] = []
        self.dataset_downloads: list[str] = []
        self._task_names_by_wf: dict[str, list[str]] = {}

    def submit(self, yaml_path, *, rsync_pairs=(), pool=None):
        wf_id = f"wf-fake-{len(self.submits) + 1}"
        parsed = _yaml.safe_load(Path(yaml_path).read_text())
        self._task_names_by_wf[wf_id] = [t["name"] for t in parsed["workflow"]["tasks"]]
        self.submits.append((yaml_path, wf_id))
        return wf_id

    def status(self, wf_id):
        self.status_calls.append(wf_id)
        names = self._task_names_by_wf.get(wf_id, [])
        return WorkflowSnapshot(
            wf_id,
            "COMPLETED",
            [TaskSnapshot(n, "COMPLETED", 0) for n in names],
        )

    def dataset_download(self, name, dest_dir):
        self.dataset_downloads.append(name)
        # ``name`` has shape ``{prefix}-{dispatch_id}-{run_id}``. The run_id
        # may itself contain hyphens (task names, framework slug, etc.) so
        # we can't reliably split. Trust the run dir in dispatch.json to
        # already exist; just stamp a manifest there.
        state = read_dispatch_state(Path(dest_dir))
        if state is None:
            return
        prefix = "odin-"
        suffix = name[len(prefix) :]
        for job in state.jobs:
            cand = f"{state.dispatch_id}-{job.run_id}"
            if suffix == cand:
                d = Path(dest_dir) / job.run_id
                d.mkdir(parents=True, exist_ok=True)
                (d / "manifest.json").write_text("{}")
                return


def test_two_bucket_dispatch_end_to_end(
    tmp_path: Path,
    cfg_path: Path,
    envs_path: Path,
    budgets_path: Path,
):
    """3 seeds × 2 envs → 6 rows; chunk_size=3 → 2 chunks → 2 workflows."""
    runs_root = tmp_path / "odin_runs"
    fakes: list[_FakeOsmoClient] = []

    def _factory(*a, **k):
        c = _FakeOsmoClient(*a, **k)
        fakes.append(c)
        return c

    with patch("tools.odin.bifrost.cli.OsmoClient", side_effect=_factory):
        rc = bifrost_cli.main(
            [
                "--osmo-config",
                str(cfg_path),
                "--physx-yaml",
                str(envs_path),
                "--budgets-yaml",
                str(budgets_path),
                "--seeds",
                "42,43,44",
                "--runs-root",
                str(runs_root),
                "--poll-interval",
                "0",
            ]
        )

    assert rc == 0
    [client] = fakes
    assert len(client.submits) == 2, "expected two OSMO workflow submits"

    # Each workflow YAML should declare the matching exec_timeout.
    timeouts_by_wf: dict[str, str] = {}
    tasks_by_wf: dict[str, list[str]] = {}
    for yaml_path, wf_id in client.submits:
        parsed = _yaml.safe_load(Path(yaml_path).read_text())
        timeouts_by_wf[wf_id] = parsed["workflow"]["timeout"]["exec_timeout"]
        tasks_by_wf[wf_id] = [t["name"] for t in parsed["workflow"]["tasks"]]
    # Ascending sort: first chunk = all 3 Cartpole rows (1800s each),
    # second = all 3 Ant rows (7200s each).
    [(_, first_wf), (_, second_wf)] = client.submits
    assert timeouts_by_wf[first_wf] == "1800s"
    assert timeouts_by_wf[second_wf] == "7200s"
    assert len(tasks_by_wf[first_wf]) == 3
    assert len(tasks_by_wf[second_wf]) == 3

    # All jobs landed in completed state.
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    assert state is not None
    assert {j.status for j in state.jobs} == {"completed"}
    assert state.osmo_workflow_ids == [first_wf, second_wf]

    agg_path = dispatch_dir / "aggregate.json"
    assert agg_path.exists()
    agg = json.loads(agg_path.read_text())
    assert agg["totals"]["runs"] == 6
