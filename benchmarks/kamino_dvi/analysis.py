# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate validated benchmark traces into five-seed summaries."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from .parsing import locate_rsl_rl_events, parse_training_trace
from .statistics import Estimate, final_window_mean, mean_ci95


@dataclass(frozen=True)
class RunMetrics:
    """Per-iteration metrics for one task, variant, and seed."""

    task: str
    variant: str
    seed: int
    num_envs: int
    iteration_time_s: tuple[float, ...]
    total_fps: tuple[float, ...]
    reward: tuple[float, ...]
    ep_length: tuple[float, ...]
    success_rate: tuple[float, ...] | None


@dataclass(frozen=True)
class VariantSummary:
    """Five-seed estimates for one task and physics variant."""

    task: str
    variant: str
    num_envs: int
    iteration_time_s: Estimate
    total_fps: Estimate
    reward: Estimate
    ep_length: Estimate
    success_rate: Estimate | None


def _steady_mean(values: tuple[float, ...]) -> float:
    if len(values) <= 10:
        raise ValueError("runtime series must contain more than ten warmup iterations")
    return statistics.mean(values[10:])


def summarize_records(records: list[RunMetrics]) -> list[VariantSummary]:
    """Reduce complete five-seed records with the approved runtime and learning windows."""
    summaries: list[VariantSummary] = []
    ordered = sorted(records, key=lambda record: (record.task, record.variant, record.seed))
    for (task, variant), grouped in groupby(ordered, key=lambda record: (record.task, record.variant)):
        runs = list(grouped)
        seeds = {run.seed for run in runs}
        counts = {run.num_envs for run in runs}
        if len(runs) != 5 or len(seeds) != 5:
            raise ValueError(f"{task}/{variant} requires five unique successful seeds")
        if len(counts) != 1:
            raise ValueError(f"{task}/{variant} mixes environment counts")
        success = None
        if all(run.success_rate is not None for run in runs):
            success = mean_ci95([final_window_mean(run.success_rate or ()) for run in runs])
        summaries.append(
            VariantSummary(
                task=task,
                variant=variant,
                num_envs=counts.pop(),
                iteration_time_s=mean_ci95([_steady_mean(run.iteration_time_s) for run in runs]),
                total_fps=mean_ci95([_steady_mean(run.total_fps) for run in runs]),
                reward=mean_ci95([final_window_mean(run.reward) for run in runs]),
                ep_length=mean_ci95([final_window_mean(run.ep_length) for run in runs]),
                success_rate=success,
            )
        )
    return summaries


def load_records(artifact_root: Path, logs_root: Path, task: str | None = None) -> list[RunMetrics]:
    """Load every completed full-run manifest and its matched TensorBoard trace."""
    records: list[RunMetrics] = []
    for manifest_path in sorted(artifact_root.glob("full__*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("identity", {})
        if manifest.get("state") != "completed" or (task is not None and identity.get("task") != task):
            continue
        bundles = tuple(manifest_path.parent.glob("benchmark_training_*.json"))
        if len(bundles) != 1:
            raise ValueError(f"{manifest_path.parent} must contain exactly one schema bundle")
        event_path = locate_rsl_rl_events(bundles[0], logs_root)
        trace = parse_training_trace(bundles[0], event_path)
        records.append(
            RunMetrics(
                task=trace.task,
                variant=str(identity["variant"]),
                seed=trace.seed,
                num_envs=trace.num_envs,
                iteration_time_s=trace.iteration_time_s,
                total_fps=trace.total_fps,
                reward=trace.reward,
                ep_length=trace.ep_length,
                success_rate=trace.success_rate,
            )
        )
    return records
