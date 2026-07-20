# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Five-seed statistics for the Kamino DVI benchmark report."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

T_975_DF4 = 2.7764451051977987


@dataclass(frozen=True)
class Estimate:
    """Arithmetic mean and two-sided 95% confidence half-width."""

    mean: float
    half_width: float
    n: int


def mean_ci95(values: Sequence[float]) -> Estimate:
    """Return the approved five-seed Student-t mean and interval."""
    samples = tuple(float(value) for value in values)
    if len(samples) != 5:
        raise ValueError("95% benchmark intervals require exactly five seeds")
    mean = statistics.mean(samples)
    half_width = T_975_DF4 * statistics.stdev(samples) / math.sqrt(len(samples))
    return Estimate(mean=mean, half_width=half_width, n=len(samples))


def final_window_mean(values: Sequence[float], window: int = 20) -> float:
    """Return the arithmetic mean over the final learning window."""
    samples = tuple(float(value) for value in values)
    if window <= 0 or len(samples) < window:
        raise ValueError("series must contain the positive final window")
    return statistics.mean(samples[-window:])


def rolling_mean(values: Sequence[float], window: int = 10) -> tuple[float, ...]:
    """Return contiguous rolling arithmetic means without padding."""
    samples = tuple(float(value) for value in values)
    if window <= 0 or len(samples) < window:
        raise ValueError("series must contain the positive rolling window")
    return tuple(statistics.mean(samples[index : index + window]) for index in range(len(samples) - window + 1))


def paired_ratio(baseline: Mapping[int, float], candidate: Mapping[int, float]) -> Estimate:
    """Return the five-seed baseline/candidate ratio paired by seed."""
    if set(baseline) != set(candidate) or len(baseline) != 5:
        raise ValueError("paired comparison requires matching five seeds")
    ratios = [float(baseline[seed]) / float(candidate[seed]) for seed in sorted(baseline)]
    return mean_ci95(ratios)
