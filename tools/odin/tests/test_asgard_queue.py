# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.jobs`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.jobs import build_queue_from_env_lists
from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list


def _env(task_id: str, framework: str = "rsl_rl", keep: bool = True, status: str = "current") -> EnvEntry:
    return EnvEntry(
        task_id=task_id,
        entry_point="isaaclab_tasks.direct.ant:AntEnv",
        env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
        group="direct/ant",
        has_rsl_rl=True,
        has_skrl=True,
        has_rl_games=False,
        framework=framework,
        num_envs=4096,
        max_iterations=300,
        keep=keep,
        status=status,
    )


def _write_env_list(tmp_path: Path, name: str, entries: list[EnvEntry]) -> Path:
    el = EnvList()
    for e in entries:
        el.groups.setdefault(e.group, []).append(e)
    out = tmp_path / name
    write_env_list(out, el, generator="test")
    return out


def test_expand_one_row_one_seed(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.task_id == "Isaac-Ant-Direct-v0"
    assert j.framework == "rsl_rl"
    assert j.backend == "physx"
    assert j.seed == 42
    assert j.num_envs == 4096
    assert j.max_iterations == 300
    assert j.run_id == "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42"
    assert j.bundle_dir_name == j.run_id
    assert j.status == "pending"
    assert j.attempts == 0


def test_expand_multiple_seeds(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=None, seeds=[42, 43, 44], dispatch_id="20260422-220000"
    )
    assert len(jobs) == 3
    assert {j.seed for j in jobs} == {42, 43, 44}
    assert len({j.run_id for j in jobs}) == 3  # all unique


def test_combines_physx_and_newton(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    newton = _write_env_list(tmp_path, "newton.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=newton, seeds=[42], dispatch_id="20260422-220000")
    assert len(jobs) == 2
    assert {j.backend for j in jobs} == {"physx", "newton"}


def test_skips_keep_false_rows(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Keep-False-v0", keep=False)],
    )
    jobs = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")
    assert [j.task_id for j in jobs] == ["Isaac-Ant-Direct-v0"]


def test_skips_stale_rows(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Stale-v0", status="stale")],
    )
    jobs = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")
    assert [j.task_id for j in jobs] == ["Isaac-Ant-Direct-v0"]


def test_include_filter_fnmatch(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Humanoid-Direct-v0"), _env("Isaac-Cartpole-Direct-v0")],
    )
    jobs = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260422-220000",
        include_filter=["Isaac-Ant-*", "Isaac-Humanoid-*"],
    )
    assert {j.task_id for j in jobs} == {"Isaac-Ant-Direct-v0", "Isaac-Humanoid-Direct-v0"}


def test_neither_yaml_raises():
    with pytest.raises(ValueError, match="at least one"):
        build_queue_from_env_lists(physx_yaml=None, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")


def test_empty_seeds_raises(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    with pytest.raises(ValueError, match="seed"):
        build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[], dispatch_id="20260422-220000")
