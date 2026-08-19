# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint reporting on the training benchmark adapters.

Pins the wiring that lets an adapter populate ``TrainingBundle.checkpoint_path``:
a hardcoded ``None`` leaves a chained play step nothing to roll out. The search
has to match every library's naming, which is why it is not a single glob.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaaclab_rl.entrypoints.common import latest_checkpoint_path

# <repo>/source/isaaclab/test/benchmark/ -> <repo>/source/isaaclab/isaaclab/...
_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / "isaaclab" / "benchmark" / "entrypoints" / "backends"


def _adapter_source(backend: str) -> str:
    return (_ADAPTER_ROOT / backend / f"benchmark_play_{backend}.py").read_text()


@pytest.mark.parametrize("backend", ("rsl_rl", "rl_games", "skrl"))
def test_training_reports_the_checkpoint_it_wrote(backend: str) -> None:
    # A chained play step reads this field; a hardcoded None leaves it nothing.
    # Either resolver satisfies that: get_checkpoint_path is the isaaclab_tasks
    # utility, latest_checkpoint_path handles the libraries whose checkpoint
    # names it does not match.
    source = (_ADAPTER_ROOT / backend / f"benchmark_train_{backend}.py").read_text()
    assert "latest_checkpoint_path(" in source or "get_checkpoint_path(" in source
    assert "checkpoint_path = None" not in source
    assert "checkpoint_path=None" not in source


def test_latest_checkpoint_path_finds_the_newest(tmp_path: Path) -> None:
    from isaaclab_rl.entrypoints.common import latest_checkpoint_path

    nested = tmp_path / "nn"
    nested.mkdir()
    (nested / "model_100.pt").write_text("x")
    older = nested / "model_100.pt"
    newer = nested / "model_200.pt"
    newer.write_text("x")
    import os

    os.utime(older, (1, 1))

    assert latest_checkpoint_path(str(tmp_path)) == str(newer.resolve())


def test_latest_checkpoint_path_is_none_when_nothing_was_written(tmp_path: Path) -> None:
    from isaaclab_rl.entrypoints.common import latest_checkpoint_path

    assert latest_checkpoint_path(str(tmp_path)) is None


@pytest.mark.parametrize(
    "filename",
    [
        "model_100.pt",  # rsl_rl
        "agent_4800.pt",  # skrl
        "best_agent.pt",  # skrl
        "Isaac-Cartpole-Direct_4800.pt",  # skrl, named after the experiment
        "last_Isaac_ep_50.pth",  # rl_games
        "model.zip",  # sb3
    ],
)
def test_latest_checkpoint_path_matches_every_library_naming(tmp_path: Path, filename: str) -> None:
    # A `model_*.pt` glob matched rsl_rl only, so every SKRL row reported no
    # checkpoint and a chained play step had nothing to roll out.
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / filename).write_text("x")

    assert latest_checkpoint_path(str(tmp_path)) == str((checkpoints / filename).resolve())
