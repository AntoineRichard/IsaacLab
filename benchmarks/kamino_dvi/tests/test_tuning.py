# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the declarative ANYmal-D Kamino DVI tuning matrix."""

from pathlib import Path

import pytest
import yaml

from benchmarks.kamino_dvi.tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    config_hash,
    hydra_overrides,
    load_tuning_matrix,
    resolve_config,
)


def test_tuning_matrix_declares_exact_anymal_protocol_and_wave1():
    """The tuning matrix must lock the approved ANYmal-D campaign."""
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    assert matrix.task == "Isaac-Velocity-Flat-AnymalD"
    assert matrix.num_envs == 4096
    assert matrix.seeds == (42, 43, 44)
    assert (matrix.preflight_iterations, matrix.screen_iterations) == (5, 40)
    assert (matrix.halve_iterations, matrix.final_iterations) == (100, 300)
    assert len(matrix.wave1) == 18
    assert all(len(candidate.overrides) == 1 for candidate in matrix.wave1)


def test_resolved_hash_is_order_independent_and_hydra_values_are_canonical():
    """Resolved configurations must hash canonically and emit canonical Hydra values."""
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = matrix.candidate("dynamics_preconditioning_true")
    resolved = resolve_config(matrix, candidate)
    assert resolved["dynamics_preconditioning"] is True
    assert config_hash(dict(reversed(tuple(resolved.items())))) == config_hash(resolved)
    assert hydra_overrides(matrix, candidate) == (
        "env.sim.physics.solver_cfg.dynamics_preconditioning=true",
    )


def _write_matrix(tmp_path: Path, mutate) -> Path:
    """Write a mutated copy of the checked-in tuning matrix."""
    data = yaml.safe_load(DEFAULT_TUNING_MATRIX_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "tuning_matrix.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda data: data["wave1"].append(data["wave1"][0]), "duplicate candidate names"),
        (lambda data: data.update(task="Isaac-Ant-Direct"), "ANYmal"),
        (lambda data: data.update(num_envs=2048), "4096"),
        (lambda data: data.update(seeds=[42, 43]), "seeds"),
        (lambda data: data["wave1"].pop(), "18"),
        (lambda data: data["wave1"][0]["overrides"].update(dvi_omega=0.5), "exactly one"),
        (lambda data: data["baseline"].pop("dvi_omega"), "baseline"),
        (lambda data: data["iterations"].update(screen=0), "positive"),
        (lambda data: data.update(learning_window=0), "positive"),
    ),
)
def test_tuning_matrix_rejects_invalid_protocol(tmp_path: Path, mutate, message: str):
    """Invalid changes to locked tuning dimensions must fail during loading."""
    with pytest.raises(ValueError, match=message):
        load_tuning_matrix(_write_matrix(tmp_path, mutate))


def test_resolve_config_rejects_unknown_override_fields():
    """Candidate fields outside the baseline schema must be rejected."""
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = type(matrix.wave1[0])("unknown", {"not_a_solver_field": 1})

    with pytest.raises(ValueError, match="unknown fields"):
        resolve_config(matrix, candidate)
