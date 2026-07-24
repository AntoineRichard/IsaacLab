# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deterministic Isaac Lab version-comparison matrix."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tools.benchmark_comparison.matrix import (
    CANARY_ATTEMPT_COUNT,
    FINAL_ATTEMPT_COUNT,
    FINAL_LOGICAL_PAIR_COUNT,
    RunSet,
    Version,
    expand_canary_matrix,
    expand_final_matrix,
    load_matrix,
)

_EXPECTED_TASK_ALIASES = {
    "cartpole": ("Isaac-Cartpole-v0", "Isaac-Cartpole"),
    "cartpole_rgb_kit": ("Isaac-Cartpole-RGB-v0", "Isaac-Cartpole-Camera"),
    "cartpole_direct": ("Isaac-Cartpole-Direct-v0", "Isaac-Cartpole-Direct"),
    "ant": ("Isaac-Ant-v0", "Isaac-Ant"),
    "ant_direct": ("Isaac-Ant-Direct-v0", "Isaac-Ant-Direct"),
    "humanoid_manager": ("Isaac-Humanoid-v0", "Isaac-Humanoid"),
    "humanoid_direct": ("Isaac-Humanoid-Direct-v0", "Isaac-Humanoid-Direct"),
    "anymal_d_flat": ("Isaac-Velocity-Flat-Anymal-D-v0", "Isaac-Velocity-Flat-AnymalD"),
    "anymal_d_rough": ("Isaac-Velocity-Rough-Anymal-D-v0", "Isaac-Velocity-Rough-AnymalD"),
    "g1_flat": ("Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Flat-G1"),
    "cassie_flat": ("Isaac-Velocity-Flat-Cassie-v0", "Isaac-Velocity-Flat-Cassie"),
    "allegro_cube": ("Isaac-Repose-Cube-Allegro-v0", "Isaac-Reorient-Cube-Allegro"),
    "franka_reach": ("Isaac-Reach-Franka-v0", "Isaac-Reach-Franka"),
}


def _write_invalid_matrix(tmp_path: Path, old: str, new: str) -> Path:
    """Write a matrix configuration with one targeted invalid substitution."""
    source_path = Path(__file__).parents[1] / "matrix.toml"
    destination_path = tmp_path / "matrix.toml"
    destination_path.write_text(source_path.read_text().replace(old, new), encoding="utf-8")
    return destination_path


def test_load_matrix_parses_explicit_task_aliases_and_run_parameters() -> None:
    """The checked-in configuration exposes every logical task and final run parameter."""
    matrix = load_matrix()

    tasks = {task.alias: task for task in matrix.tasks}

    assert {alias: (task.lab2_id, task.lab3_id) for alias, task in tasks.items()} == _EXPECTED_TASK_ALIASES
    assert tasks["cartpole_rgb_kit"].supported_modes == ("runtime-100", "runtime-1000")
    assert tasks["cartpole_rgb_kit"].enable_cameras is True
    assert tasks["cartpole_rgb_kit"].lab3_presets == ("rgb",)
    assert all(task.supported_modes is None for alias, task in tasks.items() if alias != "cartpole_rgb_kit")
    assert all(not task.enable_cameras for alias, task in tasks.items() if alias != "cartpole_rgb_kit")
    assert matrix.num_envs == 4096
    assert matrix.seeds == (42, 43, 44)
    assert [(mode.id, mode.final_bound.value, mode.final_bound.unit.value) for mode in matrix.modes] == [
        ("runtime-100", 100, "steps"),
        ("runtime-1000", 1000, "steps"),
        ("training-100", 100, "iterations"),
    ]
    assert {mode.framework for mode in matrix.modes} == {"rsl_rl"}


def test_final_matrix_expands_counterbalanced_pairs_in_deterministic_order() -> None:
    """Final runs expand to 114 pairs and 228 version attempts in the configured order."""
    expansion = expand_final_matrix(load_matrix())

    assert expansion.run_set is RunSet.FINAL
    assert len(expansion.pairs) == FINAL_LOGICAL_PAIR_COUNT == 114
    assert len(expansion.attempts) == FINAL_ATTEMPT_COUNT == 228
    assert tuple(pair.pair_order for pair in expansion.pairs) == tuple(range(114))
    assert tuple(attempt.attempt_order for attempt in expansion.attempts) == tuple(range(228))

    expected_versions = {
        42: (Version.LAB2, Version.LAB3),
        43: (Version.LAB3, Version.LAB2),
        44: (Version.LAB2, Version.LAB3),
    }
    for pair in expansion.pairs:
        assert tuple(attempt.version for attempt in pair.attempts) == expected_versions[pair.seed]
        assert tuple(attempt.version_order for attempt in pair.attempts) == (0, 1)
        for attempt in pair.attempts:
            assert (
                attempt.concrete_task
                == _EXPECTED_TASK_ALIASES[attempt.logical_task][0 if attempt.version is Version.LAB2 else 1]
            )
            assert attempt.num_envs == 4096
            assert attempt.framework == "rsl_rl"


def test_canary_matrix_has_separate_identities_and_reduced_bounds() -> None:
    """Canary attempts use the final expansion path without reusing final identities."""
    matrix = load_matrix()
    final = expand_final_matrix(matrix)
    canary = expand_canary_matrix(matrix)

    assert canary.run_set is RunSet.CANARY
    assert len(canary.pairs) == 38
    assert len(canary.attempts) == CANARY_ATTEMPT_COUNT == 76
    assert {pair.seed for pair in canary.pairs} == {42}
    assert {attempt.num_envs for attempt in canary.attempts} == {4096}
    assert {attempt.framework for attempt in canary.attempts} == {"rsl_rl"}
    assert {attempt.identity for attempt in final.attempts}.isdisjoint(attempt.identity for attempt in canary.attempts)
    assert {attempt.run_directory for attempt in final.attempts}.isdisjoint(
        attempt.run_directory for attempt in canary.attempts
    )
    assert {(attempt.mode.id, attempt.bound.value, attempt.bound.unit.value) for attempt in canary.attempts} == {
        ("runtime-100", 10, "steps"),
        ("runtime-1000", 25, "steps"),
        ("training-100", 2, "iterations"),
    }


def test_rgb_cartpole_expands_runtime_only_while_other_tasks_expand_all_modes() -> None:
    """The RGB cartpole task is limited to runtime modes without constraining locomotion."""
    expansion = expand_final_matrix(load_matrix())
    task_modes = {(pair.logical_task, pair.mode.id) for pair in expansion.pairs}

    assert ("cartpole_rgb_kit", "runtime-100") in task_modes
    assert ("cartpole_rgb_kit", "runtime-1000") in task_modes
    assert ("cartpole_rgb_kit", "training-100") not in task_modes
    assert {mode for task, mode in task_modes if task == "anymal_d_flat"} == {
        "runtime-100",
        "runtime-1000",
        "training-100",
    }
    assert {mode for task, mode in task_modes if task == "anymal_d_rough"} == {
        "runtime-100",
        "runtime-1000",
        "training-100",
    }


def test_matrix_models_are_immutable() -> None:
    """Expanded matrix data is safe to reuse across controller stages."""
    attempt = expand_final_matrix(load_matrix()).attempts[0]

    with pytest.raises(FrozenInstanceError):
        attempt.seed = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('alias = "ant"', 'alias = "cartpole"', "duplicate task alias"),
        ('lab3_id = "Isaac-Ant"', 'lab3_id = "Isaac-Cartpole"', "duplicate concrete task ID"),
        ("seeds = [42, 43, 44]", "seeds = [42, 43]", "expected 114 logical pairs"),
    ],
)
def test_load_matrix_rejects_duplicate_ids_and_incorrect_final_counts(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Invalid matrix configuration cannot create ambiguous or incomplete benchmark runs."""
    path = _write_invalid_matrix(tmp_path, old, new)

    with pytest.raises(ValueError, match=message):
        load_matrix(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'supported_modes = ["runtime-100", "runtime-1000"]',
            "supported_modes = []",
            "task supported_modes must not be empty",
        ),
        (
            'supported_modes = ["runtime-100", "runtime-1000"]',
            'supported_modes = ["runtime-100", "runtime-100"]',
            "duplicate task mode ID",
        ),
        (
            'supported_modes = ["runtime-100", "runtime-1000"]',
            'supported_modes = ["runtime-100", "runtime-999"]',
            "unknown task mode ID",
        ),
    ],
)
def test_load_matrix_rejects_invalid_task_mode_subsets(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Task mode subsets must select distinct configured benchmark modes."""
    path = _write_invalid_matrix(tmp_path, old, new)

    with pytest.raises(ValueError, match=message):
        load_matrix(path)


def test_load_matrix_rejects_incorrect_task_and_mode_shape(tmp_path: Path) -> None:
    """The fixed task-mode matrix rejects renamed modes."""
    path = _write_invalid_matrix(
        tmp_path,
        'id = "training-100"',
        'id = "training-renamed"',
    )

    with pytest.raises(ValueError, match="unexpected mode IDs"):
        load_matrix(path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('framework = "rsl_rl"', 'framework = "other_framework"'),
        ('unit = "steps"', 'unit = "iterations"'),
        ('unit = "iterations"', 'unit = "steps"'),
        ("final_bound = 1000", "final_bound = 1001"),
        ("canary_bound = 25", "canary_bound = 26"),
    ],
)
def test_load_matrix_rejects_altered_mode_parameters(tmp_path: Path, old: str, new: str) -> None:
    """Mode framework, unit, and final/canary bounds are fixed comparison parameters."""
    path = _write_invalid_matrix(tmp_path, old, new)

    with pytest.raises(ValueError, match="unexpected mode parameters"):
        load_matrix(path)
