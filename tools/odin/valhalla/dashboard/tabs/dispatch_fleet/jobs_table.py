# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A jobs section: filter row + table + inline expand rows."""

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
    expanded_run_ids: set[str] | None = None,
    ssh_tail_store: dict[str, list[str]] | None = None,
) -> html.Div:
    """Build the jobs section: filter row + table + inline expand rows.

    expanded_run_ids: which failed-row expansions are currently open.
    ssh_tail_store: keyed by run_id; values are the lines from ssh-tail.log
        (loaded on demand via the tab's load_ssh_tail callback).
    """
    jobs = dispatch_payload.get("jobs", []) or []
    expanded_run_ids = expanded_run_ids or set()
    ssh_tail_store = ssh_tail_store or {}

    if not jobs:
        return html.Div(
            id="tab-a-jobs-section-content",
            children=[
                _filter_row(status_filter, kind_filter, task_text),
                html.Div(
                    id="tab-a-jobs-empty-zero",
                    className="tab-a-empty-state",
                    children=[html.P("No jobs queued for this dispatch yet.")],
                ),
            ],
        )

    visible = filter_jobs(jobs, status_filter=status_filter, kind_filter=kind_filter, task_text=task_text)

    if not visible:
        return html.Div(
            id="tab-a-jobs-section-content",
            children=[
                _filter_row(status_filter, kind_filter, task_text),
                html.Div(
                    id="tab-a-jobs-empty",
                    className="tab-a-empty-state",
                    children=[
                        html.P("No jobs match the current filters."),
                        html.Button(
                            "Clear",
                            id="tab-a-clear-filters",
                            n_clicks=0,
                            className="tab-a-clear-button",
                        ),
                    ],
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

    body_rows: list = []
    for j in visible:
        body_rows.append(_data_row(j))
        if j.get("status") == "failed" and j.get("run_id") in expanded_run_ids:
            body_rows.append(_expand_row(j, ssh_tail_store.get(j.get("run_id"))))

    table = html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=body_rows),
        ],
    )

    return html.Div(
        id="tab-a-jobs-section-content",
        children=[_filter_row(status_filter, kind_filter, task_text), table],
    )


def _filter_row(status_filter, kind_filter, task_text):
    return html.Div(
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

    if kind:
        failure_cell = [
            html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}"),
            html.Button(
                "▸",
                id={"type": "tab-a-expand-toggle", "run_id": job.get("run_id", "")},
                n_clicks=0,
                className="tab-a-expand-toggle",
                title="Show / hide failure details",
            ),
        ]
    else:
        failure_cell = "—"

    started_ended_text = f"{started} · {ended}" if ended else (f"{started} · —" if started != "—" else "— · —")

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


def _expand_row(job: dict, ssh_tail_lines: list[str] | None) -> html.Tr:
    """Inline expansion row for a failed job: kind, attempts, message, ssh-tail."""
    failure = job.get("failure") or {}
    kind = failure.get("kind", "unknown")
    message = failure.get("message")
    attempts = int(job.get("attempts", 1) or 1)
    run_id = job.get("run_id", "")

    body: list = [
        html.Span("Kind ", className="tab-a-expand-label"),
        html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}"),
        html.Span(f"  Attempts {attempts}", className="tab-a-expand-label"),
        html.Br(),
        html.Br(),
        html.Span("Message", className="tab-a-expand-label"),
        html.Br(),
        html.Pre(
            message if message else "(no failure message recorded)",
            className="tab-a-failure-message",
        ),
        html.Button(
            "▸ Show ssh-tail.log (last 50 lines)",
            id={"type": "tab-a-ssh-tail-button", "run_id": run_id},
            n_clicks=0,
            className="tab-a-ssh-tail-button",
        ),
    ]

    if ssh_tail_lines is not None:
        if ssh_tail_lines:
            body.append(html.Pre("\n".join(ssh_tail_lines), className="tab-a-ssh-tail-pre"))
        else:
            body.append(
                html.P(
                    f"ssh-tail.log not found at {run_id}/logs/ssh-tail.log (or unreadable)",
                    className="tab-a-ssh-tail-empty",
                )
            )

    return html.Tr(
        className="tab-a-expand-row",
        children=[html.Td(colSpan=7, children=body)],
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
