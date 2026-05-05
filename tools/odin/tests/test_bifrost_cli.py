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


def test_dry_run_writes_workflow_yaml_and_exits_zero(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
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


def test_seed_expansion_creates_one_task_per_seed(
    tmp_path: Path, example_config: Path, example_physx_yaml: Path
):
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
