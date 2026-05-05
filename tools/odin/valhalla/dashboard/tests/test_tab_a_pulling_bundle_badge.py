# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A render test for the running_substate='pulling_bundle' badge."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import (
    render_jobs_section,
)


def _payload(jobs):
    return {"dispatch_id": "20260505-095154", "ended_at": None, "jobs": jobs}


def _job(running_substate=None):
    return {
        "run_id": "r1",
        "task_id": "t",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": "running",
        "assigned_to": "v1",
        "running_substate": running_substate,
    }


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def test_running_row_with_pulling_bundle_substate_renders_badge():
    section = render_jobs_section(_payload([_job(running_substate="pulling_bundle")]))
    badges = [n for n in _walk(section)
              if getattr(n, "className", "") == "tab-a-pulling-bundle-badge"]
    assert len(badges) == 1
    assert "pulling bundle" in (badges[0].children or "").lower()


def test_running_row_without_substate_does_not_render_badge():
    """Default substate (training) → no badge shown."""
    section = render_jobs_section(_payload([_job(running_substate=None)]))
    badges = [n for n in _walk(section)
              if getattr(n, "className", "") == "tab-a-pulling-bundle-badge"]
    assert badges == []


def test_running_row_with_training_substate_does_not_render_badge():
    """Explicit 'training' substate → no badge (only pulling_bundle gets one)."""
    section = render_jobs_section(_payload([_job(running_substate="training")]))
    badges = [n for n in _walk(section)
              if getattr(n, "className", "") == "tab-a-pulling-bundle-badge"]
    assert badges == []
