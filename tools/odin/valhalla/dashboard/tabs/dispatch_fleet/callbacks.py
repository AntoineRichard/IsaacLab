# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire Tab A's dcc.Interval and pattern-matching callbacks."""

from __future__ import annotations

import sys

import dash
from dash import ALL, Input, Output, State

from tools.odin.valhalla.dashboard.data import DataLayer
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.fleet_table import render_fleet_table
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.header import render_header
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import render_jobs_section

__all__ = ["register_callbacks"]


def register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Register the Tab A callbacks against the layout's slot ids."""

    @app.callback(
        Output("tab-a-header", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
    )
    def _update_header(_n, dispatch_id):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_header_children(data, dispatch_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a header callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to load dispatch.json", exc)

    @app.callback(
        Output("tab-a-fleet-table", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
    )
    def _update_fleet(_n, dispatch_id):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_fleet_children(data, dispatch_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a fleet callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render fleet table", exc)

    @app.callback(
        Output("tab-a-jobs-section", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
        Input("tab-a-status-filter", "value"),
        Input("tab-a-kind-filter", "value"),
        Input("tab-a-task-text", "value"),
        Input("tab-a-failure-filter", "data"),
        Input("tab-a-expanded-run-ids", "data"),
        Input("tab-a-ssh-tail-store", "data"),
    )
    def _update_jobs(
        _n,
        dispatch_id,
        status_filter,
        kind_filter,
        task_text,
        failure_filter,
        expanded_run_ids,
        ssh_tail_store,
    ):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_jobs_children(
                data,
                dispatch_id=dispatch_id,
                status_filter=status_filter,
                kind_filter=kind_filter,
                task_text=task_text or "",
                failure_filter=failure_filter,
                expanded_run_ids=expanded_run_ids or [],
                ssh_tail_store=ssh_tail_store or {},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a jobs callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render jobs section", exc)

    @app.callback(
        Output("tab-a-failure-filter", "data"),
        Output("tab-a-kind-filter", "value"),
        Input({"type": "tab-a-failure-pill", "kind": ALL}, "n_clicks"),
        State({"type": "tab-a-failure-pill", "kind": ALL}, "id"),
    )
    def _on_failure_pill(n_clicks_list, ids_list):
        if not n_clicks_list or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        # Pick the most-recently clicked pill: any with n_clicks > 0; the
        # last one in the list wins under Dash's pattern-matching event order.
        for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
            if n and n > 0:
                return _handle_pill_click(ident["kind"])
        return dash.no_update, dash.no_update


# -- pure helpers (testable without the Dash callback graph) ----------------------


def _compute_header_children(data: DataLayer, dispatch_id: str):
    payload = data.load_dispatch(dispatch_id)
    return render_header(payload)


def _compute_fleet_children(data: DataLayer, dispatch_id: str):
    payload = data.load_dispatch(dispatch_id)
    hardware = data.load_hardware(dispatch_id)
    return render_fleet_table(payload, hardware, data.lookup_hardware)


def _compute_jobs_children(
    data: DataLayer,
    *,
    dispatch_id: str,
    status_filter: list[str] | None,
    kind_filter: list[str] | None,
    task_text: str,
    failure_filter: str | None,
    expanded_run_ids: list[str],
    ssh_tail_store: dict[str, list[str]],
):
    payload = data.load_dispatch(dispatch_id)
    effective_kind = list(kind_filter or [])
    if failure_filter and failure_filter not in effective_kind:
        effective_kind.append(failure_filter)
    return render_jobs_section(
        payload,
        status_filter=status_filter or None,
        kind_filter=effective_kind or None,
        task_text=task_text or "",
        expanded_run_ids=set(expanded_run_ids or []),
        ssh_tail_store=ssh_tail_store or {},
    )


def _handle_pill_click(kind: str) -> tuple[str, list[str]]:
    """Return (failure-filter store value, kind-dropdown value) for a pill click."""
    return kind, [kind]


def _error_banner(message: str, exc: Exception):
    from dash import html

    return html.Div(
        className="tab-a-error-banner",
        children=[html.Strong(message), f": {type(exc).__name__}: {exc}"],
    )
