# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner per-task timeout resolution via :mod:`tools.odin.asgard.budgets`.

The bucketing system reads per-task timeouts straight from
``tools/odin/config/job_budgets.yaml`` (the same source of truth Asgard
uses), removing the redundant ``timeout_class`` indirection that lived
here before. These tests cover:

- A task explicitly listed in the budgets file gets that value.
- A task missing from ``budgets`` falls back to ``defaults.<framework>``.
- A framework that has neither an entry nor a ``defaults`` value raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.budgets import load_budgets
from tools.odin.bifrost.cli import _build_rows
from tools.odin.bifrost.config import BifrostConfigError, load_bifrost_config

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
chunk_size: 25
"""

_BUDGETS_FULL = """\
defaults:
  rsl_rl: 43200
  skrl: 43200
budgets:
  Isaac-Ant-Direct-v0:
    rsl_rl: 2100
  Isaac-Cartpole-Direct-v0:
    rsl_rl: 300
"""

_BUDGETS_NO_DEFAULT_FOR_SKRL = """\
defaults:
  rsl_rl: 43200
budgets:
  Isaac-Ant-Direct-v0:
    rsl_rl: 2100
"""


def _envs_yaml(task_id: str, framework: str) -> str:
    return (
        "groups:\n"
        "  g:\n"
        "  - task_id: " + task_id + "\n"
        "    framework: " + framework + "\n"
        "    num_envs: 4096\n"
        "    max_iterations: 500\n"
        "    keep: true\n"
    )


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_per_task_timeout_from_budgets_table(tmp_path: Path):
    """Explicit ``budgets[task][framework]`` entry wins."""
    cfg = load_bifrost_config(_write(tmp_path, "bifrost-osmo.yaml", _CFG))
    budgets = load_budgets(_write(tmp_path, "job_budgets.yaml", _BUDGETS_FULL))
    envs = _write(tmp_path, "physx.yaml", _envs_yaml("Isaac-Ant-Direct-v0", "rsl_rl"))
    rows = _build_rows(
        physx_yaml=envs,
        newton_yaml=None,
        seeds=[42],
        include_glob=None,
        dispatch_id="20260513-000000",
        cfg=cfg,
        budgets=budgets,
    )
    assert len(rows) == 1
    assert rows[0].per_task_timeout_s == 2100


def test_unknown_task_falls_back_to_framework_default(tmp_path: Path):
    """A task not in the budgets table picks up ``defaults.<framework>``."""
    cfg = load_bifrost_config(_write(tmp_path, "bifrost-osmo.yaml", _CFG))
    budgets = load_budgets(_write(tmp_path, "job_budgets.yaml", _BUDGETS_FULL))
    envs = _write(tmp_path, "physx.yaml", _envs_yaml("Isaac-Mystery-Task-v0", "skrl"))
    rows = _build_rows(
        physx_yaml=envs,
        newton_yaml=None,
        seeds=[42],
        include_glob=None,
        dispatch_id="20260513-000000",
        cfg=cfg,
        budgets=budgets,
    )
    assert len(rows) == 1
    assert rows[0].per_task_timeout_s == 43200


def test_missing_framework_default_raises(tmp_path: Path):
    """No per-task entry AND no ``defaults.<framework>`` → planner error."""
    cfg = load_bifrost_config(_write(tmp_path, "bifrost-osmo.yaml", _CFG))
    budgets = load_budgets(_write(tmp_path, "job_budgets.yaml", _BUDGETS_NO_DEFAULT_FOR_SKRL))
    envs = _write(tmp_path, "physx.yaml", _envs_yaml("Isaac-Mystery-Task-v0", "skrl"))
    with pytest.raises(BifrostConfigError, match="skrl"):
        _build_rows(
            physx_yaml=envs,
            newton_yaml=None,
            seeds=[42],
            include_glob=None,
            dispatch_id="20260513-000000",
            cfg=cfg,
            budgets=budgets,
        )


def test_task_listed_under_other_framework_falls_back_to_default(tmp_path: Path):
    """Task in budgets[X][rsl_rl] but dispatched under skrl uses skrl default."""
    cfg = load_bifrost_config(_write(tmp_path, "bifrost-osmo.yaml", _CFG))
    budgets = load_budgets(_write(tmp_path, "job_budgets.yaml", _BUDGETS_FULL))
    envs = _write(tmp_path, "physx.yaml", _envs_yaml("Isaac-Ant-Direct-v0", "skrl"))
    rows = _build_rows(
        physx_yaml=envs,
        newton_yaml=None,
        seeds=[42],
        include_glob=None,
        dispatch_id="20260513-000000",
        cfg=cfg,
        budgets=budgets,
    )
    # Ant is listed only under rsl_rl; skrl falls through to the default.
    assert rows[0].per_task_timeout_s == 43200
