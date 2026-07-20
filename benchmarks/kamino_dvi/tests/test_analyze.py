# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for report-level benchmark quality findings."""

from types import SimpleNamespace

from benchmarks.kamino_dvi.analysis import VariantSummary
from benchmarks.kamino_dvi.analyze import quality_issues
from benchmarks.kamino_dvi.statistics import Estimate


def test_quality_issues_quantifies_schema_mismatches_and_dvi_variance(tmp_path):
    """Quality findings identify known schema mismatch scope and seed sensitivity."""
    estimate = Estimate(1.0, 0.1, 3)
    summary = VariantSummary(
        "Isaac-Ant-Direct",
        "kamino_pr_dvi",
        4096,
        estimate,
        estimate,
        estimate,
        estimate,
        Estimate(0.86, 0.54, 3),
    )
    records = [SimpleNamespace(success_schema_mismatch=True) for _ in range(15)]

    issues = quality_issues(records, [summary], tmp_path)

    assert any("15 of 15 runs" in issue for issue in issues)
    assert any("seed-sensitive" in issue for issue in issues)
