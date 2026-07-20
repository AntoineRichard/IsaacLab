# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark training command construction."""

from pathlib import Path

import pytest

from benchmarks.kamino_dvi.commands import build_training_command
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import Phase, RunIdentity, TaskName, Variant


@pytest.fixture
def matrix():
    return load_matrix(DEFAULT_MATRIX_PATH)


@pytest.mark.parametrize(
    "variant,environment,preset",
    [
        (Variant.KAMINO_CURRENT, ".venv-current", "newton_kamino"),
        (Variant.KAMINO_PR_PADMM, ".venv-pr3570", "newton_kamino"),
        (Variant.KAMINO_PR_DVI, ".venv-pr3570", "newton_kamino_dvi"),
        (Variant.MJWARP, ".venv-current", "newton_mjwarp"),
        (Variant.PHYSX, ".venv-current", "physx"),
    ],
)
def test_training_command_uses_exact_environment_and_protocol(matrix, variant, environment, preset):
    """Each variant must select one interpreter and the common RSL-RL protocol."""
    repo_root = Path("/workspace/IsaacLab")
    output_path = Path("/artifacts/cartpole_seed42")
    identity = RunIdentity(
        task=TaskName.CARTPOLE,
        variant=variant,
        seed=42,
        phase=Phase.FULL,
        num_envs=4096,
        max_iterations=300,
    )

    command = build_training_command(matrix, identity, repo_root, output_path)

    assert command[:6] == [
        "/usr/bin/env",
        f"VIRTUAL_ENV={repo_root / environment}",
        f"PYTHONPATH={repo_root / 'source' / 'isaaclab_tasks'}",
        str(repo_root / "isaaclab.sh"),
        "-p",
        str(repo_root / "scripts" / "benchmarks" / "training.py"),
    ]
    assert command[6:] == [
        "--rl_library",
        "rsl_rl",
        "--task",
        "Isaac-Cartpole-Direct",
        "--num_envs",
        "4096",
        "--seed",
        "42",
        "--max_iterations",
        "300",
        "--output_path",
        str(output_path),
        "--benchmark_formatter",
        "schema",
        "--headless",
        f"presets={preset}",
    ]
    assert all(isinstance(argument, str) for argument in command)
    if variant is Variant.KAMINO_PR_DVI:
        assert not any("dynamics_solver" in argument for argument in command)
        assert not any("dynamics_preconditioning" in argument for argument in command)


def test_pr_padmm_does_not_override_newtons_default_solver(matrix):
    """The same-PR P-ADMM control must leave solver selection unset."""
    identity = RunIdentity(TaskName.ANT, Variant.KAMINO_PR_PADMM, 46, Phase.FULL, 2048, 300)

    command = build_training_command(matrix, identity, Path("/repo"), Path("/out"))

    assert not any("dynamics_solver" in argument for argument in command)
