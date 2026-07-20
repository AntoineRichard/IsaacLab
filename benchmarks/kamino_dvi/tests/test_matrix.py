# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the immutable Kamino DVI benchmark matrix."""

from pathlib import Path

import pytest

from benchmarks.kamino_dvi.matrix import (
    DEFAULT_MATRIX_PATH,
    expand_cells,
    expand_full_runs,
    expand_preflights,
    load_matrix,
    ordered_variants,
)
from benchmarks.kamino_dvi.models import EnvironmentLabel, Phase, TaskName, Variant

EXPECTED_TASKS = (
    TaskName.CARTPOLE,
    TaskName.ANT,
)
ALL_VARIANTS = (
    Variant.KAMINO_CURRENT,
    Variant.KAMINO_PR_PADMM,
    Variant.KAMINO_PR_DVI,
    Variant.MJWARP,
    Variant.PHYSX,
)


@pytest.fixture
def matrix():
    """Load the checked-in benchmark matrix."""
    return load_matrix(DEFAULT_MATRIX_PATH)


def test_matrix_has_exact_revisions_seeds_counts_and_tasks(matrix):
    """The checked-in matrix must lock every approved experiment dimension."""
    assert tuple(task.name for task in matrix.tasks) == EXPECTED_TASKS
    assert tuple(variant.name for variant in matrix.variants) == ALL_VARIANTS
    assert matrix.seeds == (42, 43, 44)
    assert matrix.environment_counts == (4096, 2048, 1024, 512, 256, 128)
    assert matrix.preflight_seed == 42
    assert matrix.preflight_iterations == 5
    assert matrix.full_iterations == 300
    assert matrix.preflight_timeout_s == 1800
    assert matrix.full_timeout_s == 14400
    assert matrix.revisions.isaaclab == "79accca281128660a786abb599f40bd335963963"
    assert matrix.revisions.schema == "47d325124080d36a270daafe3d20e2a3d11f280b"
    assert matrix.revisions.newton_current == "c7ae7c7648cd0717df39e5c94b95d5a02c997320"
    assert matrix.revisions.newton_pr == "7906676b2e5061273db96af179d7081fc6cbbba0"


def test_matrix_expands_to_10_cells_and_30_unique_full_runs(matrix):
    """Every applicable task/variant/seed identity must appear exactly once."""
    cells = expand_cells(matrix)
    full_runs = expand_full_runs(matrix)

    assert len(cells) == 10
    assert len(full_runs) == 30
    assert len(set(full_runs)) == 30
    assert all(run.phase is Phase.FULL for run in full_runs)
    assert all(run.max_iterations == 300 for run in full_runs)
    assert all(run.num_envs == 4096 for run in full_runs)


def test_preflights_cover_every_cell_at_seed_42_for_five_iterations(matrix):
    """Capacity selection must preflight each cell under one common protocol."""
    preflights = expand_preflights(matrix)

    assert len(preflights) == 10
    assert len(set(preflights)) == 10
    assert all(run.phase is Phase.PREFLIGHT for run in preflights)
    assert all(run.seed == 42 for run in preflights)
    assert all(run.max_iterations == 5 for run in preflights)
    assert all(run.num_envs == 4096 for run in preflights)
    ant_order = tuple(run.variant for run in preflights if run.task is TaskName.ANT)
    assert ant_order == tuple(reversed(ALL_VARIANTS))


@pytest.mark.parametrize("task", EXPECTED_TASKS)
def test_common_tasks_use_all_variants(matrix, task):
    """Cartpole, Ant, and ANYmal-D must include Kamino, MJWarp, and PhysX."""
    task_spec = matrix.task(task)
    assert task_spec.variants == ALL_VARIANTS


def test_variants_select_the_approved_locked_environment(matrix):
    """Only the two PR variants may execute in the candidate Newton environment."""
    environments = {variant.name: variant.environment for variant in matrix.variants}
    assert environments == {
        Variant.KAMINO_CURRENT: EnvironmentLabel.CURRENT,
        Variant.KAMINO_PR_PADMM: EnvironmentLabel.PR3570,
        Variant.KAMINO_PR_DVI: EnvironmentLabel.PR3570,
        Variant.MJWARP: EnvironmentLabel.CURRENT,
        Variant.PHYSX: EnvironmentLabel.CURRENT,
    }


def test_variant_order_rotates_by_seed_and_reverses_on_alternating_tasks(matrix):
    """Counterbalancing must be deterministic across seeds and tasks."""
    assert ordered_variants(matrix, TaskName.CARTPOLE, 42) == ALL_VARIANTS
    assert ordered_variants(matrix, TaskName.CARTPOLE, 43) == ALL_VARIANTS[1:] + ALL_VARIANTS[:1]
    assert ordered_variants(matrix, TaskName.ANT, 42) == tuple(reversed(ALL_VARIANTS))


def test_matrix_rejects_duplicate_seeds(tmp_path: Path):
    """Invalid duplicated experiment dimensions must fail before execution."""
    text = DEFAULT_MATRIX_PATH.read_text(encoding="utf-8").replace("seeds: [42, 43, 44]", "seeds: [42, 42]")
    path = tmp_path / "matrix.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate seeds"):
        load_matrix(path)
