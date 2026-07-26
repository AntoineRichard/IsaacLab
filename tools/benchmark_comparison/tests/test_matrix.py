# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deterministic Isaac Lab version-comparison matrix."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tools.benchmark_comparison.matrix import (
    CANARY_ATTEMPT_COUNT,
    CANARY_LOGICAL_PAIR_COUNT,
    FINAL_ATTEMPT_COUNT,
    FINAL_LOGICAL_PAIR_COUNT,
    RunSet,
    Version,
    expand_canary_matrix,
    expand_final_matrix,
    load_matrix,
    task_aliases_by_category,
)
from tools.benchmark_comparison.models import TaskCategory

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
    "g1_rough": ("Isaac-Velocity-Rough-G1-v0", "Isaac-Velocity-Rough-G1"),
    "cassie_flat": ("Isaac-Velocity-Flat-Cassie-v0", "Isaac-Velocity-Flat-Cassie"),
    "digit_flat": ("Isaac-Velocity-Flat-Digit-v0", "Isaac-Velocity-Flat-Digit"),
    "digit_rough": ("Isaac-Velocity-Rough-Digit-v0", "Isaac-Velocity-Rough-Digit"),
    "go1_flat": ("Isaac-Velocity-Flat-Unitree-Go1-v0", "IsaacContrib-Velocity-Flat-UnitreeGo1"),
    "go1_rough": ("Isaac-Velocity-Rough-Unitree-Go1-v0", "IsaacContrib-Velocity-Rough-UnitreeGo1"),
    "go2_flat": ("Isaac-Velocity-Flat-Unitree-Go2-v0", "Isaac-Velocity-Flat-UnitreeGo2"),
    "go2_rough": ("Isaac-Velocity-Rough-Unitree-Go2-v0", "Isaac-Velocity-Rough-UnitreeGo2"),
    "allegro_cube": ("Isaac-Repose-Cube-Allegro-v0", "Isaac-Reorient-Cube-Allegro"),
    "franka_reach": ("Isaac-Reach-Franka-v0", "Isaac-Reach-Franka"),
    "franka_cabinet_direct": ("Isaac-Franka-Cabinet-Direct-v0", "Isaac-Open-Drawer-Franka-Direct"),
    "kuka_allegro_reorient": ("Isaac-Dexsuite-Kuka-Allegro-Reorient-v0", "Isaac-Reorient-KukaAllegro"),
    "kuka_allegro_lift": ("Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "Isaac-Lift-KukaAllegro"),
}

_EXPECTED_CATEGORIES = {
    "classic": (
        "cartpole",
        "cartpole_rgb_kit",
        "cartpole_direct",
        "ant",
        "ant_direct",
        "humanoid_manager",
        "humanoid_direct",
    ),
    "locomotion_flat": (
        "anymal_d_flat",
        "g1_flat",
        "cassie_flat",
        "digit_flat",
        "go1_flat",
        "go2_flat",
    ),
    "locomotion_rough": (
        "anymal_d_rough",
        "g1_rough",
        "digit_rough",
        "go1_rough",
        "go2_rough",
    ),
    "manipulation": (
        "allegro_cube",
        "franka_reach",
        "franka_cabinet_direct",
        "kuka_allegro_reorient",
        "kuka_allegro_lift",
    ),
}

_NEW_TASK_ALIASES = (
    "g1_rough",
    "digit_flat",
    "digit_rough",
    "go1_flat",
    "go1_rough",
    "go2_flat",
    "go2_rough",
    "franka_cabinet_direct",
    "kuka_allegro_reorient",
    "kuka_allegro_lift",
)


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
    assert {task.alias: task.category for task in matrix.tasks} == {
        alias: TaskCategory(category) for category, aliases in _EXPECTED_CATEGORIES.items() for alias in aliases
    }
    category_groups = tuple(_EXPECTED_CATEGORIES.values())
    assert all(
        set(left).isdisjoint(right)
        for index, left in enumerate(category_groups)
        for right in category_groups[index + 1 :]
    )
    flat_aliases = set(_EXPECTED_CATEGORIES["locomotion_flat"])
    rough_aliases = set(_EXPECTED_CATEGORIES["locomotion_rough"])
    assert flat_aliases.isdisjoint(rough_aliases)
    assert flat_aliases | rough_aliases == {
        "anymal_d_flat",
        "anymal_d_rough",
        "g1_flat",
        "g1_rough",
        "cassie_flat",
        "digit_flat",
        "digit_rough",
        "go1_flat",
        "go1_rough",
        "go2_flat",
        "go2_rough",
    }
    assert set().union(*map(set, category_groups)) == set(_EXPECTED_TASK_ALIASES)
    assert tuple(tasks) == tuple(_EXPECTED_TASK_ALIASES)
    assert tuple(alias for aliases in category_groups for alias in aliases) != tuple(tasks)
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
    """Final runs expand to 204 pairs and 408 version attempts in the configured order."""
    expansion = expand_final_matrix(load_matrix())

    assert expansion.run_set is RunSet.FINAL
    assert len(expansion.pairs) == FINAL_LOGICAL_PAIR_COUNT == 204
    assert len(expansion.attempts) == FINAL_ATTEMPT_COUNT == 408
    assert tuple(pair.pair_order for pair in expansion.pairs) == tuple(range(204))
    assert tuple(attempt.attempt_order for attempt in expansion.attempts) == tuple(range(408))
    assert expansion.attempts[0].identity == (
        "final--cartpole--runtime-100--steps-100--seed-42--repeat-0--envs-4096--rsl_rl--lab2--version-order-0"
    )
    assert expansion.attempts[-1].identity == (
        "final--kuka_allegro_lift--training-100--iterations-100--seed-44--repeat-2--envs-4096--rsl_rl"
        "--lab3--version-order-1"
    )
    payload = "\n".join(attempt.identity for attempt in expansion.attempts).encode()
    assert hashlib.sha256(payload).hexdigest() == "8aba004dc8d09539e0fab0e8f07eb6a026f12059375a3e37e84c250c5c1c32e7"

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
    assert len(canary.pairs) == CANARY_LOGICAL_PAIR_COUNT == 68
    assert len(canary.attempts) == CANARY_ATTEMPT_COUNT == 136
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


def test_new_tasks_expand_all_modes_with_expected_final_and_canary_attempt_counts() -> None:
    """Each new task contributes all modes across the expected final and canary seeds."""
    final = expand_final_matrix(load_matrix())
    canary = expand_canary_matrix(load_matrix())

    for alias in _NEW_TASK_ALIASES:
        assert {pair.mode.id for pair in final.pairs if pair.logical_task == alias} == {
            "runtime-100",
            "runtime-1000",
            "training-100",
        }
        assert sum(attempt.logical_task == alias for attempt in final.attempts) == 18
        assert sum(attempt.logical_task == alias for attempt in canary.attempts) == 6


def test_task_aliases_by_category_filters_a_supplied_expansion_in_configured_order() -> None:
    """Category aliases retain matrix order while omitting tasks absent from an expansion."""
    aliases = task_aliases_by_category(expand_final_matrix(load_matrix()))

    assert tuple(aliases) == (
        TaskCategory.CLASSIC,
        TaskCategory.LOCOMOTION_FLAT,
        TaskCategory.LOCOMOTION_ROUGH,
        TaskCategory.MANIPULATION,
    )
    assert aliases == {
        TaskCategory.CLASSIC: _EXPECTED_CATEGORIES["classic"],
        TaskCategory.LOCOMOTION_FLAT: _EXPECTED_CATEGORIES["locomotion_flat"],
        TaskCategory.LOCOMOTION_ROUGH: _EXPECTED_CATEGORIES["locomotion_rough"],
        TaskCategory.MANIPULATION: _EXPECTED_CATEGORIES["manipulation"],
    }
    matrix_aliases = tuple(task.alias for task in load_matrix().tasks)
    for category_aliases in aliases.values():
        assert tuple(alias for alias in matrix_aliases if alias in category_aliases) == category_aliases


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
        ("seeds = [42, 43, 44]", "seeds = [42, 43]", "expected 204 logical pairs"),
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


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('category = "classic"\n', "", "task.category must be a string"),
        ('category = "classic"', 'category = "unknown"', "'unknown' is not a valid TaskCategory"),
        ('alias = "ant"', 'alias = "cartpole"', "duplicate task alias"),
    ],
)
def test_load_matrix_rejects_missing_unknown_and_duplicate_task_categories(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Every configured alias has one recognized category assignment."""
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
