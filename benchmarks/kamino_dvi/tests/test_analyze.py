# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for report-level benchmark quality findings."""

from types import SimpleNamespace

from benchmarks.kamino_dvi.analysis import VariantSummary
from benchmarks.kamino_dvi.analyze import quality_issues
from benchmarks.kamino_dvi.statistics import Estimate


def test_quality_issues_quantifies_schema_mismatches_per_task(tmp_path):
    """Schema findings separate task counts and distinguish value mismatches from missing data."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [VariantSummary("Ant", "kamino_current", 4096, estimate, estimate, estimate, estimate, estimate)]
    records = [
        SimpleNamespace(
            task="Ant", success_schema_mismatch=True, success_schema_mismatch_points=2, success_rate=(1,) * 3
        ),
        SimpleNamespace(
            task="Ant", success_schema_mismatch=False, success_schema_mismatch_points=0, success_rate=(1,) * 3
        ),
        SimpleNamespace(
            task="Cartpole", success_schema_mismatch=True, success_schema_mismatch_points=1, success_rate=(1,) * 3
        ),
        SimpleNamespace(
            task="ANYmal-D", success_schema_mismatch=False, success_schema_mismatch_points=0, success_rate=(1,) * 3
        ),
    ]

    issues = quality_issues(records, summaries, tmp_path)

    assert any("Ant" in issue and "1/2 runs" in issue and "2/6 points" in issue for issue in issues)
    assert any("Cartpole" in issue and "1/1 runs" in issue and "1/3 points" in issue for issue in issues)
    assert any("ANYmal-D" in issue and "0/1 runs" in issue and "0/3 points" in issue for issue in issues)
    assert any("every required reward, episode-length, and success field exists" in issue for issue in issues)
    assert any("value mismatch, not missing data" in issue for issue in issues)


def test_quality_issues_reports_seed_sensitive_learning_for_any_variant(tmp_path):
    """A large success interval is a weak-learning warning independent of solver variant."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [
        VariantSummary("ANYmal-D", "kamino_current", 4096, estimate, estimate, estimate, estimate, estimate),
        VariantSummary("ANYmal-D", "mjwarp", 4096, estimate, estimate, estimate, estimate, Estimate(0.75, 1.07, 3)),
    ]

    issues = quality_issues([], summaries, tmp_path)

    assert any("ANYmal-D MJWarp" in issue and "seed-sensitive weak learning" in issue for issue in issues)
    assert any("not a runtime or stability failure" in issue for issue in issues)
