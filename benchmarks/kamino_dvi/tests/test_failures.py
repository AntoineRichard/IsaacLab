# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark failure classification."""

from pathlib import Path

import pytest

from benchmarks.kamino_dvi.failures import classify_failure
from benchmarks.kamino_dvi.models import FailureCategory, RetryLineage

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "logs"


@pytest.mark.parametrize(
    "log",
    [
        "cuda_oom.txt",
        "allocation.txt",
        "contact_capacity.txt",
    ],
)
def test_capacity_logs_are_classified_for_environment_fallback(log):
    """Only explicit memory/contact capacity evidence may lower environment count."""
    text = (_FIXTURE_ROOT / log).read_text(encoding="utf-8")
    failure = classify_failure(1, False, 0, 300, True, "", text, RetryLineage())
    assert failure.category is FailureCategory.CAPACITY


@pytest.mark.parametrize("log", ["non_finite.txt", "divergence.txt", "nan.txt"])
def test_numerical_logs_are_retained_at_selected_count(log):
    """Numerical failures must remain distinct from capacity failures."""
    text = (_FIXTURE_ROOT / log).read_text(encoding="utf-8")
    failure = classify_failure(1, False, 73, 300, True, text, "", RetryLineage(attempt=1, parent_run_id="x"))
    assert failure.category is FailureCategory.NUMERICAL
    assert failure.completed_iterations == 73
    assert failure.retry.attempt == 1


@pytest.mark.parametrize(
    "returncode,timed_out,completed,artifact,expected",
    [
        (None, True, 22, True, FailureCategory.TIMEOUT),
        (2, False, 22, True, FailureCategory.CRASH),
        (0, False, 299, True, FailureCategory.INCOMPLETE),
        (0, False, 300, False, FailureCategory.ARTIFACT),
    ],
)
def test_non_capacity_failures_have_unambiguous_primary_category(returncode, timed_out, completed, artifact, expected):
    """Timeout, crash, incomplete, and artifact failures must not overlap."""
    failure = classify_failure(returncode, timed_out, completed, 300, artifact, "", "", RetryLineage())
    assert failure.category is expected


def test_failure_record_retains_signal_exception_and_last_200_lines():
    """Failure diagnostics must remain compact without dropping the useful tail."""
    lines = [f"line {index}" for index in range(250)] + ["ValueError: broken run"]
    failure = classify_failure(-9, False, 2, 300, True, "\n".join(lines), "", RetryLineage())

    assert failure.category is FailureCategory.CRASH
    assert failure.signal == 9
    assert failure.exception_type == "ValueError"
    assert len(failure.log_tail) == 200
    assert failure.log_tail[0] == "line 51"
    assert failure.log_tail[-1] == "ValueError: broken run"
