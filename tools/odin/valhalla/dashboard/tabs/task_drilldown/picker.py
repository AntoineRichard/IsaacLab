# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab B picker — searchable single dropdown of (task, framework, backend) rows."""

from __future__ import annotations

from dash import dcc, html

from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

__all__ = ["list_row_options", "render_picker"]


def list_row_options(aggregate_payload: dict) -> list[dict]:
    """Return ``dcc.Dropdown`` options for every (task, framework, backend) row.

    Sorted A-Z by task name.
    """
    rows = aggregate_payload.get("rows", []) or []
    options = []
    for row in rows:
        task = str(row.get("task", ""))
        framework = str(row.get("framework", ""))
        backend = str(row.get("backend", ""))
        options.append(
            {
                "label": f"{task} · {framework} × {backend}",
                "value": f"{task}|{framework}|{backend}",
            }
        )
    options.sort(key=lambda o: o["label"])
    return options


def render_picker(aggregate_payload: dict, selected: TaskSelection | None) -> html.Div:
    """Build the picker Div.

    Returns Div(id='tab-b-picker') containing a searchable dcc.Dropdown
    'tab-b-row-select'. The dropdown's value is the pipe-separated row key;
    `selected` is resolved to a value if present, else left as None.
    """
    options = list_row_options(aggregate_payload)
    selected_value: str | None = None
    if selected is not None and selected.task and selected.framework and selected.backend:
        candidate = f"{selected.task}|{selected.framework}|{selected.backend}"
        if any(o["value"] == candidate for o in options):
            selected_value = candidate
    return html.Div(
        id="tab-b-picker",
        className="tab-b-picker-row",
        children=[
            html.Span("Row", className="tab-b-picker-label"),
            dcc.Dropdown(
                id="tab-b-row-select",
                options=options,
                value=selected_value,
                searchable=True,
                placeholder="Pick a (task × framework × backend) row…",
                className="tab-b-picker-dropdown",
            ),
        ],
    )
