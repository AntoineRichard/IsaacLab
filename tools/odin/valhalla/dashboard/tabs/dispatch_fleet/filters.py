# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure data filter for the Tab A jobs table."""

from __future__ import annotations

__all__ = ["filter_jobs"]


def filter_jobs(
    jobs: list[dict],
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
) -> list[dict]:
    """Apply the three filters in sequence.

    - ``status_filter``: empty / None = pass through. Otherwise keep jobs whose
      ``status`` is in the list.
    - ``kind_filter``: empty / None = pass through. Otherwise keep jobs whose
      ``failure.kind`` is in the list (which implicitly excludes non-failed jobs).
    - ``task_text``: empty string = pass through. Otherwise keep jobs whose
      ``task_id`` contains the text (case-insensitive substring).

    All three are AND-combined.
    """
    needle = (task_text or "").lower()
    out: list[dict] = []
    for job in jobs:
        if status_filter and job.get("status") not in status_filter:
            continue
        if kind_filter:
            failure = job.get("failure") or {}
            if failure.get("kind") not in kind_filter:
                continue
        if needle:
            task_id = str(job.get("task_id", "")).lower()
            if needle not in task_id:
                continue
        out.append(job)
    return out
