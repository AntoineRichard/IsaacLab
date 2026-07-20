# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict schema-bundle and TensorBoard parsing for Kamino DVI runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class MissingBenchmarkFieldError(ValueError):
    """Raised when the benchmark stack omitted a required metric field."""


@dataclass(frozen=True)
class TrainingTrace:
    """Aligned per-iteration metrics for one completed training run."""

    task: str
    seed: int
    num_envs: int
    iterations: int
    iteration_time_s: tuple[float, ...]
    collection_fps: tuple[float, ...]
    total_fps: tuple[float, ...]
    reward: tuple[float, ...]
    ep_length: tuple[float, ...]
    success_rate: tuple[float, ...]
    success_schema_mismatch: bool
    resources: dict[str, Any]
    success_schema_mismatch_points: int = 0


def _field(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise MissingBenchmarkFieldError(path)
        value = value[component]
    return value


def _series(data: dict[str, Any], path: str, iterations: int) -> tuple[float, ...]:
    value = _field(data, path)
    if not isinstance(value, list) or len(value) != iterations:
        raise MissingBenchmarkFieldError(f"{path} must contain {iterations} values")
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values):
        raise MissingBenchmarkFieldError(f"{path} contains non-finite values")
    return values


def _tb_series(accumulator: EventAccumulator, tag: str, iterations: int) -> tuple[float, ...]:
    if tag not in accumulator.Tags().get("scalars", []):
        raise MissingBenchmarkFieldError(f"TensorBoard:{tag}")
    events = accumulator.Scalars(tag)
    steps = tuple(event.step for event in events)
    expected_steps = tuple(range(iterations))
    if steps != expected_steps:
        raise MissingBenchmarkFieldError(f"TensorBoard:{tag} steps {steps!r} != {expected_steps!r}")
    values = tuple(float(event.value) for event in events)
    if not all(math.isfinite(item) for item in values):
        raise MissingBenchmarkFieldError(f"TensorBoard:{tag} contains non-finite values")
    return values


def locate_rsl_rl_events(bundle_path: Path, logs_root: Path) -> Path:
    """Locate the RSL-RL event file matching a bundle task and UTC start time."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    task = str(_field(bundle, "run.task"))
    started = datetime.fromisoformat(str(_field(bundle, "run.start_time_utc")))
    matches: list[tuple[float, Path]] = []
    for run_path in logs_root.glob("rsl_rl/*/*/run.json"):
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run.get("task") != task:
                continue
            delta = abs((datetime.fromisoformat(str(run["created_at"])) - started).total_seconds())
            events = tuple(run_path.parent.glob("events.out.tfevents.*"))
            if len(events) == 1:
                matches.append((delta, events[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not matches or min(matches)[0] > 2.0:
        raise MissingBenchmarkFieldError(f"TensorBoard events for {task} at {started.isoformat()}")
    return min(matches)[1]


def parse_training_trace(bundle_path: Path, event_path: Path) -> TrainingTrace:
    """Parse and align one completed schema v1.1 bundle and TensorBoard trace."""
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.1":
        raise ValueError(f"expected schema version 1.1, got {data.get('schema_version')!r}")
    if _field(data, "run.status") != "completed":
        raise ValueError("training bundle is not completed")
    iterations = int(_field(data, "runtime.iterations_completed"))
    if iterations != int(_field(data, "run.max_iterations")):
        raise ValueError("completed iterations do not match max iterations")

    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    collection_time = _tb_series(accumulator, "Perf/collection_time", iterations)
    learning_time = _tb_series(accumulator, "Perf/learning_time", iterations)
    total_fps = _tb_series(accumulator, "Perf/total_fps", iterations)
    steps_per_iteration = int(_field(data, "runtime.steps_per_iteration"))
    iteration_time = tuple(collection + learning for collection, learning in zip(collection_time, learning_time))
    collection_fps = tuple(steps_per_iteration / collection for collection in collection_time)

    reward = _series(data, "learning.reward.series_per_iter", iterations)
    ep_length = _series(data, "learning.ep_length.series_per_iter", iterations)
    schema_success = _series(data, "learning.success_rate.series_per_iter", iterations)
    success = _tb_series(accumulator, "Metrics/success_rate", iterations)
    mismatch_points = sum(
        not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7) for left, right in zip(schema_success, success)
    )
    return TrainingTrace(
        task=str(_field(data, "run.task")),
        seed=int(_field(data, "run.seed")),
        num_envs=int(_field(data, "run.num_envs")),
        iterations=iterations,
        iteration_time_s=iteration_time,
        collection_fps=collection_fps,
        total_fps=total_fps,
        reward=reward,
        ep_length=ep_length,
        success_rate=success,
        success_schema_mismatch=mismatch_points > 0,
        resources=dict(_field(data, "resources")),
        success_schema_mismatch_points=mismatch_points,
    )
