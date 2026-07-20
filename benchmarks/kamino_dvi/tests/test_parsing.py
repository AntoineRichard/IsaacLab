# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for schema and TensorBoard benchmark trace parsing."""

import json
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from benchmarks.kamino_dvi.parsing import (
    MissingBenchmarkFieldError,
    _series,
    locate_rsl_rl_events,
    parse_training_trace,
)


def _write_bundle(path: Path, *, include_reward: bool = True) -> None:
    learning = {
        "ep_length": {"series_per_iter": [10.0, 20.0, 30.0]},
        "success_rate": {"series_per_iter": [1.0, 0.7, 0.4]},
    }
    if include_reward:
        learning["reward"] = {"series_per_iter": [1.0, 2.0, 3.0]}
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "run": {
                    "status": "completed",
                    "task": "Isaac-Cartpole-Direct",
                    "seed": 42,
                    "num_envs": 4096,
                    "max_iterations": 3,
                },
                "runtime": {"iterations_completed": 3, "steps_per_iteration": 65536},
                "resources": {"gpu_mem_gb": {"mean": 1.2, "std": 0.1, "peak": 1.3}},
                "learning": learning,
            }
        ),
        encoding="utf-8",
    )


def _write_events(path: Path, *, success_values: tuple[float, ...] | None = (1.0, 0.5, 0.0)) -> None:
    writer = SummaryWriter(path)
    for step, (collection, learning, fps) in enumerate(
        [(0.2, 0.1, 200_000.0), (0.1, 0.05, 400_000.0), (0.08, 0.04, 500_000.0)]
    ):
        writer.add_scalar("Perf/collection_time", collection, step)
        writer.add_scalar("Perf/learning_time", learning, step)
        writer.add_scalar("Perf/total_fps", fps, step)
        if success_values is not None:
            writer.add_scalar("Metrics/success_rate", success_values[step], step)
    writer.close()


def test_parse_training_trace_aligns_runtime_and_learning_series(tmp_path: Path):
    """Runtime comes from TensorBoard and learning curves retain exact iteration alignment."""
    bundle = tmp_path / "bundle.json"
    events = tmp_path / "events"
    _write_bundle(bundle)
    _write_events(events)

    trace = parse_training_trace(bundle, events)

    assert trace.iteration_time_s == pytest.approx((0.3, 0.15, 0.12))
    assert trace.collection_fps == pytest.approx((327680.0, 655360.0, 819200.0))
    assert trace.total_fps == pytest.approx((200000.0, 400000.0, 500000.0))
    assert trace.reward == (1.0, 2.0, 3.0)
    assert trace.ep_length == (10.0, 20.0, 30.0)
    assert trace.success_rate == (1.0, 0.5, 0.0)
    assert trace.success_schema_mismatch is True


def test_parse_training_trace_reports_missing_required_schema_field(tmp_path: Path):
    """A missing reward curve identifies a benchmark-stack bug instead of becoming zero."""
    bundle = tmp_path / "bundle.json"
    events = tmp_path / "events"
    _write_bundle(bundle, include_reward=False)
    _write_events(events)

    with pytest.raises(MissingBenchmarkFieldError, match="learning.reward.series_per_iter"):
        parse_training_trace(bundle, events)


def test_parse_training_trace_requires_schema_success_series(tmp_path: Path):
    """Completed bundles must include schema success data for every iteration."""
    bundle = tmp_path / "bundle.json"
    events = tmp_path / "events"
    _write_bundle(bundle)
    data = json.loads(bundle.read_text(encoding="utf-8"))
    del data["learning"]["success_rate"]
    bundle.write_text(json.dumps(data), encoding="utf-8")
    _write_events(events)

    with pytest.raises(MissingBenchmarkFieldError, match="learning.success_rate.series_per_iter"):
        parse_training_trace(bundle, events)


def test_parse_training_trace_rejects_non_finite_schema_success(tmp_path: Path):
    """Non-finite schema success data must not enter aggregation."""
    bundle = tmp_path / "bundle.json"
    events = tmp_path / "events"
    _write_bundle(bundle)
    data = json.loads(bundle.read_text(encoding="utf-8"))
    data["learning"]["success_rate"]["series_per_iter"] = [1.0, float("nan"), 0.0]
    bundle.write_text(json.dumps(data), encoding="utf-8")
    _write_events(events)

    with pytest.raises(MissingBenchmarkFieldError, match="learning.success_rate.series_per_iter.*non-finite"):
        parse_training_trace(bundle, events)


def test_parse_training_trace_rejects_non_finite_tensorboard_success(tmp_path: Path):
    """Non-finite TensorBoard success data must not enter aggregation."""
    bundle = tmp_path / "bundle.json"
    events = tmp_path / "events"
    _write_bundle(bundle)
    _write_events(events, success_values=(1.0, float("nan"), 0.0))

    with pytest.raises(MissingBenchmarkFieldError, match="TensorBoard:Metrics/success_rate.*non-finite"):
        parse_training_trace(bundle, events)


def test_parse_training_trace_uses_schema_success_without_tensorboard_tag(tmp_path: Path):
    """Required schema success remains available when TensorBoard omits the tag."""
    bundle = tmp_path / "bundle.json"
    events = tmp_path / "events"
    _write_bundle(bundle)
    _write_events(events, success_values=None)

    trace = parse_training_trace(bundle, events)

    assert trace.success_rate == (1.0, 0.7, 0.4)
    assert trace.success_schema_mismatch is False


def test_series_rejects_non_finite_metric_values():
    """A completed bundle with NaN learning data must not enter aggregation."""
    data = {"learning": {"reward": {"series_per_iter": [1.0, float("nan"), 3.0]}}}

    with pytest.raises(MissingBenchmarkFieldError, match="non-finite"):
        _series(data, "learning.reward.series_per_iter", 3)


def test_locate_rsl_rl_events_matches_task_and_utc_creation_time(tmp_path: Path):
    """Bundle provenance must resolve the exact RSL-RL event directory."""
    bundle = tmp_path / "bundle.json"
    _write_bundle(bundle)
    data = json.loads(bundle.read_text(encoding="utf-8"))
    data["run"]["start_time_utc"] = "2026-07-20T10:52:22.008473+00:00"
    bundle.write_text(json.dumps(data), encoding="utf-8")
    matching = tmp_path / "logs" / "rsl_rl" / "cartpole" / "2026-07-20_12-52-22"
    matching.mkdir(parents=True)
    (matching / "run.json").write_text(
        json.dumps({"created_at": "2026-07-20T10:52:22.092279+00:00", "task": "Isaac-Cartpole-Direct"}),
        encoding="utf-8",
    )
    event = matching / "events.out.tfevents.test"
    event.touch()

    assert locate_rsl_rl_events(bundle, tmp_path / "logs") == event
