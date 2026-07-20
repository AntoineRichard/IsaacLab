# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic benchmark failure classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import FailureCategory, RetryLineage

_CAPACITY_PATTERN = re.compile(
    r"out of memory|outofmemoryerror|allocation failed|max_contacts_per_world.*(?:capacity|exceed)", re.IGNORECASE
)
_NUMERICAL_PATTERN = re.compile(r"non[- ]finite|\bnan\b|solver divergence|divergence detected", re.IGNORECASE)
_EXCEPTION_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):")


@dataclass(frozen=True)
class FailureRecord:
    """Compact diagnostics for one failed benchmark run."""

    category: FailureCategory
    returncode: int | None
    signal: int | None
    exception_type: str | None
    completed_iterations: int
    log_tail: tuple[str, ...]
    retry: RetryLineage


def classify_failure(
    returncode: int | None,
    timed_out: bool,
    completed_iterations: int,
    expected_iterations: int,
    artifact_present: bool,
    stdout: str,
    stderr: str,
    retry: RetryLineage,
) -> FailureRecord:
    """Classify one unsuccessful run using the approved precedence."""
    combined = "\n".join(part for part in (stdout, stderr) if part)
    if _CAPACITY_PATTERN.search(combined):
        category = FailureCategory.CAPACITY
    elif timed_out:
        category = FailureCategory.TIMEOUT
    elif _NUMERICAL_PATTERN.search(combined):
        category = FailureCategory.NUMERICAL
    elif returncode not in (None, 0):
        category = FailureCategory.CRASH
    elif completed_iterations < expected_iterations:
        category = FailureCategory.INCOMPLETE
    elif not artifact_present:
        category = FailureCategory.ARTIFACT
    else:
        raise ValueError("successful run cannot be classified as a failure")

    exceptions = _EXCEPTION_PATTERN.findall(combined)
    signal = -returncode if returncode is not None and returncode < 0 else None
    return FailureRecord(
        category=category,
        returncode=returncode,
        signal=signal,
        exception_type=exceptions[-1] if exceptions else None,
        completed_iterations=completed_iterations,
        log_tail=tuple(combined.splitlines()[-200:]),
        retry=retry,
    )
