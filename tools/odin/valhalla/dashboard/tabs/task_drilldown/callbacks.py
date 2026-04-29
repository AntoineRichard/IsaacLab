# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire Tab B's callbacks against the layout."""

from __future__ import annotations

import sys

import dash
from dash import Input, Output, State

from tools.odin.valhalla.dashboard.data import DataLayer
from tools.odin.valhalla.dashboard.tabs.task_drilldown.picker import render_picker
from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import (
    TaskSelection,
    parse_query_string,
    serialize,
)

__all__ = ["register_callbacks"]


def register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Register Tab B's callbacks (3 standard + 2 update; update lands in T10)."""

    @app.callback(
        Output("tab-b-picker", "children"),
        Input("tab-b-dispatch-id", "data"),
        State("url", "search"),
    )
    def _init_picker(dispatch_id, search):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_picker_children(data, dispatch_id, search or "")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b init_picker: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render picker", exc)

    @app.callback(
        Output("tab-b-selection", "data"),
        Input("url", "search"),
    )
    def _sync_url(search):
        return _sync_url_to_selection(search or "")

    @app.callback(
        Output("url", "search"),
        Input("tab-b-row-select", "value"),
        State("url", "search"),
    )
    def _picker_to_url(value, current_search):
        new_search = _serialize_to_url(value)
        if new_search == (current_search or ""):
            return dash.no_update
        return new_search


# -- pure helpers -----------------------------------------------------------


def _compute_picker_children(data: DataLayer, dispatch_id: str, search: str):
    aggregate = data.load_aggregate(dispatch_id)
    if aggregate is None:
        from dash import html

        return html.Div(
            id="tab-b-picker",
            className="tab-b-error-banner",
            children=[
                html.Strong("Aggregate not yet generated for this dispatch"),
                " — Tab B is empty until aggregation completes.",
            ],
        )
    selected = parse_query_string(search)
    return render_picker(aggregate, selected)


def _sync_url_to_selection(search: str) -> str | None:
    sel = parse_query_string(search)
    if sel.task and sel.framework and sel.backend:
        return f"{sel.task}|{sel.framework}|{sel.backend}"
    return None


def _serialize_to_url(value: str | None) -> str:
    if not value:
        return ""
    parts = value.split("|", 2)
    if len(parts) != 3:
        return ""
    sel = TaskSelection(parts[0] or None, parts[1] or None, parts[2] or None)
    return serialize(sel)


def _error_banner(message: str, exc: Exception):
    from dash import html

    return html.Div(
        className="tab-b-error-banner",
        children=[html.Strong(message), f": {type(exc).__name__}: {exc}"],
    )
