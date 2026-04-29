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
    """Register Tab B's callbacks (3 standard + 4 update; update lands in T10)."""

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

    @app.callback(
        Output("tab-b-curves", "children"),
        Output("tab-b-stats", "children"),
        Input("tab-b-selection", "data"),
        Input("tab-b-dispatch-id", "data"),
    )
    def _update_curves_and_stats(selection_value, dispatch_id):
        if not dispatch_id:
            return dash.no_update, dash.no_update
        try:
            return _compute_curves_and_stats(data, dispatch_id, selection_value)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b curves/stats: {type(exc).__name__}: {exc}", file=sys.stderr)
            banner = _error_banner("Failed to render curves/stats", exc)
            return banner, banner

    @app.callback(
        Output("tab-b-trend", "children"),
        Input("tab-b-selection", "data"),
        Input("tab-b-dispatch-id", "data"),
        Input("tab-b-trend-metric", "data"),
        Input("tab-b-trend-mode", "data"),
    )
    def _update_trend(selection_value, dispatch_id, metric, mode):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_trend_children(data, dispatch_id, selection_value, metric, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b trend: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render trend section", exc)

    @app.callback(
        Output("tab-b-trend-metric", "data"),
        Input("tab-b-trend-metric-select", "value"),
    )
    def _on_trend_metric(value):
        return value or dash.no_update

    @app.callback(
        Output("tab-b-trend-mode", "data"),
        Input("tab-b-trend-mode-toggle", "value"),
    )
    def _on_trend_mode(value):
        return value or dash.no_update


# -- pure helpers -----------------------------------------------------------


def _compute_picker_children(data: DataLayer, dispatch_id: str, search: str):
    aggregate = data.load_aggregate(dispatch_id)
    if aggregate is None:
        from dash import html

        return html.Div(
            id="tab-b-picker-content",
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


def _compute_curves_and_stats(data, dispatch_id: str, selection_value: str | None):
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.curves import render_curves
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.stats import render_stats_panel

    aggregate = data.load_aggregate(dispatch_id)
    if aggregate is None:
        return (
            _banner_div(
                "Aggregate not yet generated for this dispatch — Tab B is empty until aggregation completes.",
                id_="tab-b-curves-content",
            ),
            _banner_div("", id_="tab-b-stats-content"),
        )

    if not selection_value:
        return (
            _banner_div("Pick a row from the dropdown above.", id_="tab-b-curves-content"),
            _banner_div("", id_="tab-b-stats-content"),
        )

    parts = selection_value.split("|", 2)
    if len(parts) != 3:
        return (
            _banner_div("Selection malformed.", id_="tab-b-curves-content"),
            _banner_div("", id_="tab-b-stats-content"),
        )
    task, framework, backend = parts

    row = next(
        (
            r
            for r in aggregate.get("rows", []) or []
            if r.get("task") == task and r.get("framework") == framework and r.get("backend") == backend
        ),
        None,
    )
    if row is None:
        return (
            _banner_div(
                f"Row not found in this dispatch: {task} · {framework} × {backend}. "
                "Pick another row from the dropdown.",
                id_="tab-b-curves-content",
            ),
            _banner_div("", id_="tab-b-stats-content"),
        )

    seeds = row.get("seeds") or {}
    bundles: dict[str, dict] = {}
    for seed_key, seed in seeds.items():
        run_id = seed.get("run_id")
        if not run_id:
            continue
        training = data.load_training(dispatch_id, run_id)
        if training is not None:
            bundles[seed_key] = training

    divergent = row.get("divergent_seeds", []) or []
    return (
        render_curves(bundles, divergent_seeds=divergent, heading=task),
        render_stats_panel(row),
    )


def _compute_trend_children(data, dispatch_id: str, selection_value: str | None, metric: str, mode: str):
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section

    if not selection_value:
        return _banner_div("Pick a row from the dropdown to populate the trend.", id_="tab-b-trend-content")
    parts = selection_value.split("|", 2)
    if len(parts) != 3:
        return _banner_div("Selection malformed.", id_="tab-b-trend-content")
    task, framework, backend = parts
    selection = TaskSelection(task or None, framework or None, backend or None)
    return render_trend_section(
        data,
        current_dispatch_id=dispatch_id,
        selection=selection,
        metric=metric,
        mode=mode,
    )


def _banner_div(message: str, *, id_: str = "tab-b-banner"):
    from dash import html

    return html.Div(
        id=id_,
        className="tab-b-empty-state",
        children=[html.P(message)],
    )


def _error_banner(message: str, exc: Exception):
    from dash import html

    return html.Div(
        className="tab-b-error-banner",
        children=[html.Strong(message), f": {type(exc).__name__}: {exc}"],
    )
