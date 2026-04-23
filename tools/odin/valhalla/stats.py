# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stats helpers for the Valhalla aggregator.

Exposes :class:`Stats` (the mean/std/min/max/cv_pct blob used in
``aggregate.json`` rows) plus :func:`stats_over` to compute one over a
list of samples, and :func:`is_divergent` to flag outlier seeds via a
z-score threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Stats", "stats_over", "is_divergent"]


@dataclass
class Stats:
    """Cross-seed stats blob emitted inside each row's ``aggregate`` block."""

    mean: float
    std: float
    min: float
    max: float
    cv_pct: float


def stats_over(values: list[float]) -> Stats:
    """Compute the aggregate stats over ``values``.

    Args:
        values: Completed-seed samples. Must be non-empty.

    Returns:
        :class:`Stats` with population std for ``len(values) == 2`` and
        sample std (ddof=1) for ``len(values) >= 3``. ``len(values) == 1``
        yields ``std=0.0``, ``cv_pct=0.0``.

    Raises:
        ValueError: When ``values`` is empty.
    """
    if not values:
        raise ValueError("stats_over requires at least one sample, got empty list")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    elif n == 2:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)  # population, ddof=0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))  # sample, ddof=1
    cv_pct = 0.0 if mean == 0.0 else 100.0 * std / abs(mean)
    return Stats(mean=mean, std=std, min=min(values), max=max(values), cv_pct=cv_pct)


def is_divergent(values: list[float], z: float) -> list[int]:
    """Return the indices of values farther than ``z`` standard deviations from the mean.

    Uses sample std (ddof=1). Returns an empty list for ``len(values) < 3``
    — two-sample outlier detection is meaningless.

    Args:
        values: Per-seed metric samples.
        z: Threshold in standard-deviation multiples.

    Returns:
        Sorted list of indices of offending samples (empty if none).
    """
    if len(values) < 3:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    if std == 0.0:
        return []
    return [i for i, v in enumerate(values) if abs(v - mean) > z * std]
