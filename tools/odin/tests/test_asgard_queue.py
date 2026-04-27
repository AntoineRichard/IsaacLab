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


def _env(
    task_id: str,
    framework: str = "rsl_rl",
    keep: bool = True,
    status: str = "current",
    presets_available: list[str] | None = None,
    native_backend: str | None = None,
) -> EnvEntry:
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
        presets_available=list(presets_available) if presets_available is not None else [],
        native_backend=native_backend,
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
    jobs, _ = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")
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
    jobs, _ = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=None, seeds=[42, 43, 44], dispatch_id="20260422-220000"
    )
    assert len(jobs) == 3
    assert {j.seed for j in jobs} == {42, 43, 44}
    assert len({j.run_id for j in jobs}) == 3  # all unique


def test_combines_physx_and_newton(tmp_path: Path):
    physx = _write_env_list(tmp_path, "physx.yaml", [_env("Isaac-Ant-Direct-v0")])
    newton = _write_env_list(tmp_path, "newton.yaml", [_env("Isaac-Ant-Direct-v0")])
    jobs, _ = build_queue_from_env_lists(
        physx_yaml=physx, newton_yaml=newton, seeds=[42], dispatch_id="20260422-220000"
    )
    assert len(jobs) == 2
    assert {j.backend for j in jobs} == {"physx", "newton"}


def test_skips_keep_false_rows(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Keep-False-v0", keep=False)],
    )
    jobs, _ = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")
    assert [j.task_id for j in jobs] == ["Isaac-Ant-Direct-v0"]


def test_skips_stale_rows(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Stale-v0", status="stale")],
    )
    jobs, _ = build_queue_from_env_lists(physx_yaml=physx, newton_yaml=None, seeds=[42], dispatch_id="20260422-220000")
    assert [j.task_id for j in jobs] == ["Isaac-Ant-Direct-v0"]


def test_include_filter_fnmatch(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0"), _env("Isaac-Humanoid-Direct-v0"), _env("Isaac-Cartpole-Direct-v0")],
    )
    jobs, _ = build_queue_from_env_lists(
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


def test_supported_pair_produces_jobs(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Ant-Direct-v0", presets_available=["physx", "newton"])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42, 43, 44],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 3
    assert skipped == []


def test_unsupported_pair_skips_with_telemetry(tmp_path: Path):
    # Empty list = unknown → pass through (not skipped). Use [newton] to
    # exercise the actual skip path against the physx backend.
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-NewtonOnly-v0", presets_available=["newton"])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42, 43, 44],
        dispatch_id="20260427-100000",
    )
    assert jobs == []
    assert len(skipped) == 3
    s = skipped[0]
    assert s.task_id == "Isaac-NewtonOnly-v0"
    assert s.framework == "rsl_rl"
    assert s.backend == "physx"
    assert s.seed in {42, 43, 44}
    assert s.reason == "preset_unsupported"
    assert s.presets_available == ["newton"]


def test_empty_presets_available_passes_through(tmp_path: Path):
    """Unknown preset support (empty list) must NOT trigger the skip path."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Unknown-Presets-v0", presets_available=[])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 1
    assert skipped == []


def test_dual_preset_supports_both_backends(tmp_path: Path):
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Dual-v0", presets_available=["physx", "newton"])],
    )
    newton = _write_env_list(
        tmp_path,
        "newton.yaml",
        [_env("Isaac-Dual-v0", presets_available=["physx", "newton"])],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=newton,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 2
    assert {j.backend for j in jobs} == {"physx", "newton"}
    assert skipped == []


def test_include_filter_runs_before_preset_filter(tmp_path: Path):
    """Rows excluded by --include must NOT appear in skipped[]."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [
            _env("Isaac-Ant-Direct-v0", presets_available=["physx"]),
            _env("Isaac-NewtonOnly-v0", presets_available=["newton"]),
        ],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
        include_filter=["Isaac-Ant-*"],
    )
    assert len(jobs) == 1
    assert jobs[0].task_id == "Isaac-Ant-Direct-v0"
    assert skipped == []  # NewtonOnly filtered out before preset gate


def test_native_mismatch_skips_with_telemetry(tmp_path: Path):
    """presets_available=[] AND native_backend != requested → skipped with reason='native_backend_mismatch'."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Quadcopter-Direct-v0", presets_available=[], native_backend="physx")],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=None,
        newton_yaml=physx,
        seeds=[42, 43, 44],
        dispatch_id="20260427-100000",
    )
    assert jobs == []
    assert len(skipped) == 3
    s = skipped[0]
    assert s.task_id == "Isaac-Quadcopter-Direct-v0"
    assert s.framework == "rsl_rl"
    assert s.backend == "newton"
    assert s.reason == "native_backend_mismatch"
    assert s.presets_available == []
    assert s.native_backend == "physx"


def test_native_match_passes_through_to_runtime(tmp_path: Path):
    """presets_available=[] AND native_backend == requested → JobEntry created (runtime injects)."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Quadcopter-Direct-v0", presets_available=[], native_backend="physx")],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 1
    assert skipped == []


def test_unknown_native_passes_through(tmp_path: Path):
    """presets_available=[] AND native_backend=None (truly unknown) → JobEntry created (runtime catches)."""
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Unknown-Native-v0", presets_available=[], native_backend=None)],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert len(jobs) == 1
    assert skipped == []


def test_preset_unsupported_takes_precedence_over_native(tmp_path: Path):
    """When BOTH rules could fire (presets_available populated + native mismatch), rule 1 wins.

    Theoretically a task could have a PresetCfg with no physx alternative
    but default to physx internally.  Rule 1 (preset_unsupported) is the
    correct classification — the task explicitly opts INTO the preset
    system and excludes physx from the menu.
    """
    physx = _write_env_list(
        tmp_path,
        "physx.yaml",
        [_env("Isaac-Edge-v0", presets_available=["newton"], native_backend="physx")],
    )
    jobs, skipped = build_queue_from_env_lists(
        physx_yaml=physx,
        newton_yaml=None,
        seeds=[42],
        dispatch_id="20260427-100000",
    )
    assert jobs == []
    assert len(skipped) == 1
    assert skipped[0].reason == "preset_unsupported"
    # native_backend telemetry still populated for rule 1 entries.
    assert skipped[0].native_backend == "physx"
