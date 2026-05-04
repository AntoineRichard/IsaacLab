# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A jobs section: filter row + table + inline expand rows."""

from __future__ import annotations

from datetime import datetime, timezone

from dash import dcc, html

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.filters import filter_jobs

__all__ = ["render_filter_row", "render_jobs_rows", "render_jobs_section"]


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
    running_tail_shown: set[str] | None = None,
    running_tail_store: dict[str, dict] | None = None,
    retry_queue: set[str] | None = None,
) -> html.Div:
    """Build the jobs section: filter row + table + inline expand rows.

    expanded_run_ids: which failed-row expansions are currently open.
    ssh_tail_store: keyed by run_id; values are the lines from ssh-tail.log
        (loaded on demand via the tab's load_ssh_tail callback).
    retry_queue: set of run_ids the operator has tagged for retry. Drives
        the per-row toggle highlight + the banner above the table.
    """
    jobs = dispatch_payload.get("jobs", []) or []
    expanded_run_ids = expanded_run_ids or set()
    ssh_tail_store = ssh_tail_store or {}
    running_tail_shown = running_tail_shown or set()
    running_tail_store = running_tail_store or {}
    retry_queue = retry_queue or set()

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

    dispatch_id = str(dispatch_payload.get("dispatch_id", "") or "")

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
        body_rows.append(_data_row(j, dispatch_id, retry_queue))
        if j.get("status") == "failed" and j.get("run_id") in expanded_run_ids:
            body_rows.append(_expand_row(j, ssh_tail_store.get(j.get("run_id"))))
        if j.get("status") == "running" and j.get("run_id") in running_tail_shown:
            body_rows.append(_expand_running_row(j, running_tail_store.get(j.get("run_id"))))

    table = html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=body_rows),
        ],
    )

    section_children = [_filter_row(status_filter, kind_filter, task_text)]
    banner = _retry_banner(dispatch_id, retry_queue)
    if banner is not None:
        section_children.append(banner)
    section_children.append(table)
    return html.Div(id="tab-a-jobs-section-content", children=section_children)


def render_filter_row(status_filter=None, kind_filter=None, task_text=""):
    """Public alias of :func:`_filter_row` — used by the static layout so the
    filter components exist at cold mount (callbacks reference their values
    by id; without this they wouldn't be in the DOM yet)."""
    return _filter_row(status_filter, kind_filter, task_text)


def render_jobs_rows(
    dispatch_payload: dict,
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
    expanded_run_ids: set[str] | None = None,
    ssh_tail_store: dict[str, list[str]] | None = None,
    running_tail_shown: set[str] | None = None,
    running_tail_store: dict[str, dict] | None = None,
    retry_queue: set[str] | None = None,
):
    """Return just the rows portion (table-or-empty) of the jobs section.

    Callable used by the live ``update_jobs`` callback — the filter row is
    static (rendered once in the layout) so it doesn't get re-rendered each
    tick (which would wipe filter state).
    """
    jobs = dispatch_payload.get("jobs", []) or []
    expanded_run_ids = expanded_run_ids or set()
    ssh_tail_store = ssh_tail_store or {}
    running_tail_shown = running_tail_shown or set()
    running_tail_store = running_tail_store or {}
    retry_queue = retry_queue or set()

    if not jobs:
        return html.Div(
            id="tab-a-jobs-empty-zero",
            className="tab-a-empty-state",
            children=[html.P("No jobs queued for this dispatch yet.")],
        )

    visible = filter_jobs(jobs, status_filter=status_filter, kind_filter=kind_filter, task_text=task_text)

    if not visible:
        return html.Div(
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
        )

    dispatch_id = str(dispatch_payload.get("dispatch_id", "") or "")

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
        body_rows.append(_data_row(j, dispatch_id, retry_queue))
        if j.get("status") == "failed" and j.get("run_id") in expanded_run_ids:
            body_rows.append(_expand_row(j, ssh_tail_store.get(j.get("run_id"))))
        if j.get("status") == "running" and j.get("run_id") in running_tail_shown:
            body_rows.append(_expand_running_row(j, running_tail_store.get(j.get("run_id"))))
    table = html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=body_rows),
        ],
    )
    banner = _retry_banner(dispatch_id, retry_queue)
    if banner is None:
        return table
    return html.Div(children=[banner, table])


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


def _retry_banner(dispatch_id: str, retry_queue: set[str]) -> html.Div | None:
    """Top-of-jobs banner shown only when one or more rows are tagged for retry.

    Renders the operator's exact ``odin-dispatch --resume … --retry-failed=<csv>``
    command in a copy-paste-friendly box.
    """
    if not retry_queue:
        return None
    csv = ",".join(sorted(retry_queue))
    cmd = (
        "PYTHONPATH=. python3 -u -m tools.odin.asgard.cli "
        "--fleet fleet.yaml --physx-yaml tools/odin/config/physx_envs.yaml "
        "--seeds 42,43,44 "
        f"--resume {dispatch_id} --retry-failed={csv} --verbose"
    )
    return html.Div(
        id="tab-a-retry-banner",
        className="tab-a-retry-banner",
        children=[
            html.Div(
                className="tab-a-retry-banner-header",
                children=[
                    html.Strong(f"{len(retry_queue)} job(s) tagged for retry"),
                    html.Span(
                        " — live runners consume these automatically; use the command below after dispatch end.",
                        className="tab-a-retry-banner-hint",
                    ),
                ],
            ),
            html.Pre(cmd, className="tab-a-retry-banner-cmd"),
        ],
    )


def _data_row(job: dict, dispatch_id: str, retry_queue: set[str] | None = None) -> html.Tr:
    status = str(job.get("status", "unknown"))
    failure = job.get("failure") or {}
    kind = failure.get("kind")
    attempts = int(job.get("attempts", 1) or 1)
    host = job.get("assigned_to") or "—"
    started = _relative_time(job.get("started_at"))
    ended = _relative_time(job.get("ended_at"))
    retry_queue = retry_queue or set()
    run_id = job.get("run_id", "")

    status_children = [
        html.Span(status.capitalize(), className=f"tab-a-pill tab-a-job-status-{status}"),
    ]
    if attempts > 1:
        status_children.append(html.Span(f"×{attempts}", className="tab-a-attempts-badge"))

    # Running jobs get the 👁 tail-toggle adjacent to the status pill — the
    # tail isn't a failure detail, so the failure column is the wrong home.
    if status == "running":
        status_children.append(
            html.Button(
                "👁",
                id={"type": "tab-a-running-tail-toggle", "run_id": run_id},
                n_clicks=0,
                className="tab-a-expand-toggle tab-a-running-tail-toggle",
                title="Show / hide running stdout tail",
            )
        )

    if kind:
        is_queued = run_id in retry_queue
        retry_btn = html.Button(
            "✓" if is_queued else "↻",
            id={"type": "tab-a-retry-toggle", "run_id": run_id},
            n_clicks=0,
            className="tab-a-retry-toggle" + (" tab-a-retry-toggle-queued" if is_queued else ""),
            title=("Remove from retry queue" if is_queued else "Tag for live retry if the runner is active"),
        )
        failure_cell = [
            html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}"),
            html.Button(
                "▸",
                id={"type": "tab-a-expand-toggle", "run_id": run_id},
                n_clicks=0,
                className="tab-a-expand-toggle",
                title="Show / hide failure details",
            ),
            retry_btn,
        ]
    else:
        failure_cell = "—"

    started_ended_text = f"{started} · {ended}" if ended else (f"{started} · —" if started != "—" else "— · —")

    task_id = job.get("task_id", "")
    framework = job.get("framework", "")
    backend = job.get("backend", "")
    task_link = dcc.Link(
        task_id,
        href=f"/{dispatch_id}/task-drilldown?task={task_id}&framework={framework}&backend={backend}",
        className="tab-a-task-link",
    )

    return html.Tr(
        children=[
            html.Td(task_link),
            html.Td(f"{framework} × {backend}", className="tab-a-mono"),
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


def _expand_running_row(job: dict, tail_entry: dict | None) -> html.Tr:
    """Inline expansion row for a running job's stdout tail."""
    run_id = job.get("run_id", "")
    tail_entry = tail_entry or {}
    source = tail_entry.get("source") or "stdout tail"
    fetched_at = tail_entry.get("fetched_at")
    warning = tail_entry.get("warning")
    lines = tail_entry.get("lines")

    body: list = [
        html.Button(
            "Refresh",
            id={"type": "tab-a-running-tail-refresh", "run_id": run_id},
            n_clicks=0,
            className="tab-a-ssh-tail-button tab-a-running-tail-refresh",
        ),
        html.Span(source, className="tab-a-running-tail-source"),
    ]
    if fetched_at:
        body.extend([html.Span("  Fetched ", className="tab-a-expand-label"), html.Span(fetched_at)])

    if warning:
        body.append(
            html.P(
                f"Running stdout tail unavailable: {warning}",
                className="tab-a-ssh-tail-empty tab-a-running-tail-warning",
            )
        )
    elif lines is None:
        body.append(html.P("Running stdout tail not loaded yet.", className="tab-a-ssh-tail-empty"))
    elif lines:
        body.append(html.Pre("\n".join(lines), className="tab-a-ssh-tail-pre"))
    else:
        body.append(html.P("Running stdout tail is empty or unavailable.", className="tab-a-ssh-tail-empty"))

    return html.Tr(
        className="tab-a-expand-row tab-a-running-tail-row",
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
