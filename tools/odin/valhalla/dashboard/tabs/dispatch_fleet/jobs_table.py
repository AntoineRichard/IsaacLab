# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A jobs section: filter row + table.

Inline expansion + ssh-tail rendering land in Task 7.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dash import dcc, html

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.filters import filter_jobs

__all__ = ["render_jobs_section"]


_STATUS_OPTIONS = [
    {"label": "Pending", "value": "pending"},
    {"label": "Running", "value": "running"},
    {"label": "Completed", "value": "completed"},
    {"label": "Failed", "value": "failed"},
]

_KIND_OPTIONS = [
    {"label": "hugin_crash", "value": "hugin_crash"},
    {"label": "gpu_lost", "value": "gpu_lost"},
    {"label": "preset_unsupported", "value": "preset_unsupported"},
    {"label": "timeout", "value": "timeout"},
    {"label": "infrastructure", "value": "infrastructure"},
    {"label": "hugin_malformed_bundle", "value": "hugin_malformed_bundle"},
]


def render_jobs_section(
    dispatch_payload: dict,
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
) -> html.Div:
    """Build the jobs section: filter row + filtered table.

    Spec 1 Task 6 — no expand row support yet (added in Task 7).
    """
    jobs = dispatch_payload.get("jobs", []) or []
    visible = filter_jobs(jobs, status_filter=status_filter, kind_filter=kind_filter, task_text=task_text)

    filter_row = html.Div(
        className="tab-a-jobs-filter-row",
        children=[
            html.Span("Status", className="tab-a-filter-label"),
            dcc.Dropdown(
                id="tab-a-status-filter",
                options=_STATUS_OPTIONS,
                value=status_filter or [],
                multi=True,
                placeholder="All",
                className="tab-a-filter-dropdown",
            ),
            html.Span("Failure", className="tab-a-filter-label"),
            dcc.Dropdown(
                id="tab-a-kind-filter",
                options=_KIND_OPTIONS,
                value=kind_filter or [],
                multi=True,
                placeholder="All",
                className="tab-a-filter-dropdown",
            ),
            html.Span("Task", className="tab-a-filter-label"),
            dcc.Input(
                id="tab-a-task-text",
                type="text",
                value=task_text,
                placeholder="filter task…",
                debounce=True,
                className="tab-a-filter-input",
            ),
        ],
    )

    header = html.Tr(
        children=[
            html.Th("Task"),
            html.Th("Framework × Backend"),
            html.Th("Seed"),
            html.Th("Status"),
            html.Th("Failure"),
            html.Th("Host"),
            html.Th("Started / Ended"),
        ]
    )
    rows = [_data_row(j) for j in visible]

    table = html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=rows),
        ],
    )

    return html.Div(
        id="tab-a-jobs-section-content",
        children=[filter_row, table],
    )


def _data_row(job: dict) -> html.Tr:
    status = str(job.get("status", "unknown"))
    failure = job.get("failure") or {}
    kind = failure.get("kind")
    attempts = int(job.get("attempts", 1) or 1)
    host = job.get("assigned_to") or "—"
    started = _relative_time(job.get("started_at"))
    ended = _relative_time(job.get("ended_at"))

    status_children = [
        html.Span(status.capitalize(), className=f"tab-a-pill tab-a-job-status-{status}"),
    ]
    if attempts > 1:
        status_children.append(html.Span(f"×{attempts}", className="tab-a-attempts-badge"))

    failure_cell = (
        html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}") if kind else "—"
    )

    started_ended_text = (
        f"{started} · {ended}" if ended else (f"{started} · —" if started != "—" else "— · —")
    )

    return html.Tr(
        children=[
            html.Td(job.get("task_id", "")),
            html.Td(f"{job.get('framework', '')} × {job.get('backend', '')}", className="tab-a-mono"),
            html.Td(str(job.get("seed", ""))),
            html.Td(status_children),
            html.Td(failure_cell),
            html.Td(host, className="tab-a-mono"),
            html.Td(started_ended_text, className="tab-a-muted"),
        ]
    )


def _relative_time(ts: str | None) -> str:
    """Return a human-readable relative time, e.g. ``'3m ago'``. ``None`` → ``'—'``."""
    if not ts:
        return "—"
    try:
        # Strip a trailing Z for fromisoformat compatibility on Python 3.10.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    now = datetime.now(timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
