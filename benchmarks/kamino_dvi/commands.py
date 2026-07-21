# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure command construction for Kamino DVI training runs."""

from pathlib import Path

from .environment import python_executable
from .models import BenchmarkMatrix, RunIdentity


def build_training_command(
    matrix: BenchmarkMatrix,
    identity: RunIdentity,
    repo_root: Path,
    output_path: Path,
) -> list[str]:
    """Build one shell-free unified RSL-RL benchmark command.

    Args:
        matrix: Validated benchmark matrix.
        identity: Exact run identity to execute.
        repo_root: Isaac Lab worktree root.
        output_path: Directory for the canonical benchmark bundle.

    Returns:
        Subprocess argument vector for the run.
    """
    variant = matrix.variant(identity.variant)
    environment_root = python_executable(repo_root, variant.environment).parent.parent
    command = [
        "/usr/bin/env",
        f"VIRTUAL_ENV={environment_root}",
        (f"PYTHONPATH={repo_root / 'source' / 'isaaclab_newton'}:{repo_root / 'source' / 'isaaclab_tasks'}"),
        str(repo_root / "isaaclab.sh"),
        "-p",
        str(repo_root / "scripts" / "benchmarks" / "training.py"),
        "--rl_library",
        "rsl_rl",
        "--task",
        identity.task.value,
        "--num_envs",
        str(identity.num_envs),
        "--seed",
        str(identity.seed),
        "--max_iterations",
        str(identity.max_iterations),
        "--output_path",
        str(output_path),
        "--benchmark_formatter",
        "schema",
        "--headless",
        f"presets={variant.preset}",
    ]
    if variant.dynamics_solver is not None:
        command.extend(
            [
                f"env.sim.physics.solver_cfg.dynamics_solver={variant.dynamics_solver}",
                "env.sim.physics.solver_cfg.dynamics_preconditioning=False",
            ]
        )
    return command
