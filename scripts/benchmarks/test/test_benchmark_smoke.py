# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke tests for training and playing benchmarked policies."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-v0"
_PLAY_BUNDLE_KEYS = {"run", "versions", "hardware", "runtime", "resources"}


def _load_play_bundle(output_path: Path) -> dict:
    """Load the schema play bundle from an output directory."""
    data = json.loads(next(output_path.glob("*_schema.json")).read_text())
    assert data.keys() >= _PLAY_BUNDLE_KEYS
    return data


def _run(command: list[str], working_directory: Path) -> None:
    """Run a benchmark command and report its trailing output on failure."""
    result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        pytest.fail(
            f"{Path(command[2]).name} rc={result.returncode}\n"
            f"STDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )


@pytest.mark.parametrize(
    (
        "library",
        "num_envs",
        "max_iterations",
        "formatter",
        "expect_reward_series",
        "expect_success_rate",
        "expect_checkpoint",
    ),
    [
        ("rl_games", 16, 20, "schema,json", True, False, False),
        ("rsl_rl", 16, 2, "schema,json", True, False, False),
        ("sb3", 16, 70, "schema,json", False, False, True),
        ("skrl", 16, 20, "schema,json", True, False, False),
    ],
)
def test_training_and_play_write_bundles(
    tmp_path,
    library: str,
    num_envs: int,
    max_iterations: int,
    formatter: str,
    expect_reward_series: bool,
    expect_success_rate: bool,
    expect_checkpoint: bool,
):
    """Each RL library trains and plays a policy with benchmark output."""
    training_output = tmp_path / "training"
    play_output = tmp_path / "play"
    common_args = [
        "--rl_library",
        library,
        "--task",
        _TASK,
        "--num_envs",
        str(num_envs),
        "presets=physx",
        "--headless",
        "--benchmark_formatter",
        formatter,
    ]

    _run(
        [
            str(ROOT / "isaaclab.sh"),
            "-p",
            str(ROOT / "scripts" / "benchmarks" / "training.py"),
            *common_args[:6],
            "--max_iterations",
            str(max_iterations),
            *common_args[6:],
            "--output_path",
            str(training_output),
        ],
        tmp_path,
    )
    training_data = json.loads(next(training_output.glob("*_schema.json")).read_text())
    assert training_data["schema_version"] == "1.0"
    assert training_data["run"]["config"]["physics_backend"] == "physx"
    assert training_data["runtime"]["startup_time_s"]["python_imports"] > 0
    assert training_data["runtime"]["startup_time_s"]["task_config"] > 0
    assert 1 <= training_data["runtime"]["iterations_completed"] <= max_iterations
    assert training_data["run"]["framework"] == library
    assert training_data["runtime"]["total_fps"]["mean"] > 0
    assert training_data["learning"]["reward"]["series_per_iter"] is not None
    assert training_data["learning"]["reward"]["final_ema"] is not None
    if expect_reward_series:
        assert len(training_data["learning"]["reward"]["series_per_iter"]) >= 1
    if expect_success_rate:
        assert training_data["success_rate"] is not None
    if expect_checkpoint:
        assert Path(training_data["checkpoint_path"]).is_file()

    _run(
        [
            str(ROOT / "isaaclab.sh"),
            "-p",
            str(ROOT / "scripts" / "benchmarks" / "play.py"),
            *common_args,
            "--num_frames",
            "250",
            "--checkpoint",
            "latest",
            "--output_path",
            str(play_output),
        ],
        tmp_path,
    )
    play_data = _load_play_bundle(play_output)
    assert play_data["run"]["framework"] == library
    assert play_data["runtime"]["total_fps"]["mean"] > 0
    assert play_data["checkpoint_path"]
    assert play_data["reward"] is not None
    assert "mean" in play_data["reward"]

    training_json = json.loads(next(training_output.glob("*_json.json")).read_text())
    play_json = json.loads(next(play_output.glob("*_json.json")).read_text())
    assert isinstance(training_json, list) and training_json
    assert isinstance(play_json, list) and play_json
