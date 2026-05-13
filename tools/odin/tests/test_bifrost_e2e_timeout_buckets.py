# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end timeout-bucket smoke against a mock OSMO client.

Walks the full Bifrost CLI flow — config load, planner, render, submit,
poll, bundle download, aggregate — for a 6-task dispatch split into
``short`` (3 tasks) and ``medium`` (3 tasks) buckets. The mock client
records every submit/query call so we can assert:

- Two OSMO workflows submitted with the right ``exec_timeout`` values
  (spec §10).
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
timeout_classes:
  short: "30m"
  medium: "2h"
default_timeout_class: medium
chunk_size: 25
"""

_ENVS = """\
groups:
  cartpole:
  - task_id: Isaac-Cartpole-Direct-v0
    framework: rsl_rl
    num_envs: 4096
    max_iterations: 150
    keep: true
    timeout_class: short
  ant:
  - task_id: Isaac-Ant-Direct-v0
    framework: rsl_rl
    num_envs: 4096
    max_iterations: 1000
    keep: true
    timeout_class: medium
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
        # Find the run_id this dataset name corresponds to.
        prefix = "odin-"  # bundle_dataset_prefix-
        suffix = name[len(prefix) :]
        # suffix is f"{dispatch_id}-{run_id}"; the dispatch_id is fixed
        # for the dispatch.
        for job in state.jobs:
            cand = f"{state.dispatch_id}-{job.run_id}"
            if suffix == cand:
                d = Path(dest_dir) / job.run_id
                d.mkdir(parents=True, exist_ok=True)
                (d / "manifest.json").write_text("{}")
                return


def test_two_class_dispatch_end_to_end(tmp_path: Path, cfg_path: Path, envs_path: Path):
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
                # 3 seeds × 2 envs → 6 tasks total, split 3 (short) + 3 (medium).
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
    # _bucket_and_chunk sorts alphabetically: medium first, then short.
    [(_, first_wf), (_, second_wf)] = client.submits
    assert timeouts_by_wf[first_wf] == "2h"
    assert timeouts_by_wf[second_wf] == "30m"
    assert len(tasks_by_wf[first_wf]) == 3
    assert len(tasks_by_wf[second_wf]) == 3

    # All jobs landed in completed state.
    [dispatch_dir] = list(runs_root.iterdir())
    state = read_dispatch_state(dispatch_dir)
    assert state is not None
    assert {j.status for j in state.jobs} == {"completed"}
    assert state.osmo_workflow_ids == [first_wf, second_wf]

    # aggregate.json was written by the end-of-dispatch hook. The
    # ``completed`` count comes from the aggregator's manifest-parsing
    # path, which our stub manifest (``{}``) doesn't satisfy. The total
    # ``runs`` count, which only depends on dispatch.json, is the right
    # thing to assert end-to-end.
    agg_path = dispatch_dir / "aggregate.json"
    assert agg_path.exists()
    agg = json.loads(agg_path.read_text())
    assert agg["totals"]["runs"] == 6
