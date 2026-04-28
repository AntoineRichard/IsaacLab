# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plotly Dash app factory for the Odin dashboard.

Builds the SPA shell: header (logo + dispatch dropdown + live/done pill),
tab strip, page-content area. Routing lives in :func:`route_pathname` so it
can be unit-tested without the live Dash callback machinery.
"""

from __future__ import annotations

import re
from pathlib import Path

import dash
from dash import Input, Output, dcc, html

from tools.odin.valhalla.dashboard.data import DataLayer

__all__ = ["create_app", "route_pathname"]


_DISPATCH_ID_RE = re.compile(r"^\d{8}-\d{6}$")
_TAB_IDS = {"dispatch-fleet", "task-drilldown", "startup"}


def create_app(runs_root: Path, initial_dispatch: Path | None = None) -> dash.Dash:
    """Build the Dash app. Pure factory — no global state."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Odin"
    data = DataLayer(runs_root)
    app.layout = _build_layout(initial_dispatch)
    _register_callbacks(app, data)
    return app


def _build_layout(initial_dispatch: Path | None) -> html.Div:
    initial_path = "/"
    if initial_dispatch is not None:
        initial_path = f"/{initial_dispatch.name}/dispatch-fleet"
    return html.Div(
        id="app-root",
        children=[
            dcc.Location(id="url", refresh=False, pathname=initial_path),
            dcc.Store(id="active-dispatch", storage_type="memory"),
            html.Div(id="page-content"),
        ],
    )


def _register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def _on_url(pathname: str):
        return route_pathname(pathname or "/", data)

    _register_tab_callbacks(app, data)


def _register_tab_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Walk the three known tab module names; call register(app, data) if present.

    Spec 0's placeholder has no register(); Specs 1/2/3 add modules that wire
    their dcc.Interval / pattern-matching callbacks at app startup. Importing
    a missing module is silently OK — that just means the tab spec hasn't
    landed yet.
    """
    import importlib

    for module_name in (
        "tools.odin.valhalla.dashboard.tabs.dispatch_fleet",
        "tools.odin.valhalla.dashboard.tabs.task_drilldown",
        "tools.odin.valhalla.dashboard.tabs.startup",
    ):
        try:
            tab_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        register_fn = getattr(tab_module, "register", None)
        if register_fn is not None:
            register_fn(app, data)


def route_pathname(pathname: str, data: DataLayer):
    """Map a URL pathname to the Dash component tree to render at /page-content.

    Pulled out as a free function so unit tests can drive routing without
    spinning up the Dash callback graph.
    """
    parts = [p for p in pathname.split("/") if p]
    if not parts:
        return _landing(data)
    dispatch_id = parts[0]
    if not _DISPATCH_ID_RE.match(dispatch_id):
        return _not_found(pathname)
    # Verify the dispatch actually exists.
    try:
        data.load_dispatch(dispatch_id)
    except FileNotFoundError:
        return _not_found(pathname)
    if len(parts) == 1:
        # /<id>/ → redirect to default tab
        return html.Div(
            id="redirect-to-tab-a",
            children=[
                dcc.Location(id="redirect-loc", href=f"/{dispatch_id}/dispatch-fleet", refresh=True),
            ],
        )
    tab_id = parts[1]
    if tab_id not in _TAB_IDS:
        return _not_found(pathname)
    return _render_tab(dispatch_id, tab_id, data)


def _landing(data: DataLayer) -> html.Div:
    """Multi-dispatch landing: real table of dispatches sorted newest-first."""
    summaries = data.list_dispatches()
    if not summaries:
        return html.Div(
            id="landing-root",
            children=[
                html.H2("Odin dashboard"),
                html.P("No dispatches under runs_root yet. Run odin-dispatch to create one."),
            ],
        )
    header = html.Tr(
        children=[
            html.Th("Dispatch"),
            html.Th("Started"),
            html.Th("Ended"),
            html.Th("Total"),
            html.Th("Completed"),
            html.Th("Failed"),
            html.Th("Pending"),
            html.Th("Skipped"),
            html.Th("Hosts"),
        ],
    )
    rows = [
        html.Tr(
            children=[
                html.Td(html.A(s.dispatch_id, href=f"/{s.dispatch_id}/")),
                html.Td(s.started_at or "—"),
                html.Td(s.ended_at or "—"),
                html.Td(str(s.jobs_total)),
                html.Td(str(s.jobs_completed)),
                html.Td(str(s.jobs_failed)),
                html.Td(str(s.jobs_pending)),
                html.Td(str(s.skipped_total)),
                html.Td(", ".join(s.hostnames) or "—"),
            ],
        )
        for s in summaries
    ]
    return html.Div(
        id="landing-root",
        children=[
            html.H2("Odin dashboard"),
            html.Table(children=[html.Thead(children=[header]), html.Tbody(children=rows)]),
        ],
    )


def _not_found(pathname: str) -> html.Div:
    return html.Div(
        id="not-found-root",
        children=[
            html.H2("Not found"),
            html.P(f"No route for {pathname!r}."),
            dcc.Link("Back to dashboard", href="/"),
        ],
    )


def _render_tab(dispatch_id: str, tab_id: str, data: DataLayer) -> html.Div:
    """Render the tab body for /<id>/<tab_id>.

    Looks for a real tab module under ``tools.odin.valhalla.dashboard.tabs``
    matching ``tab_id``; falls back to the placeholder when the module is
    absent. Specs 1/2/3 add their modules; Spec 0 only ships ``_placeholder``.
    """
    import importlib

    module_name = {
        "dispatch-fleet": "tools.odin.valhalla.dashboard.tabs.dispatch_fleet",
        "task-drilldown": "tools.odin.valhalla.dashboard.tabs.task_drilldown",
        "startup": "tools.odin.valhalla.dashboard.tabs.startup",
    }.get(tab_id)
    if module_name is not None:
        try:
            tab_module = importlib.import_module(module_name)
            if hasattr(tab_module, "render"):
                return tab_module.render(dispatch_id, tab_id)
        except ModuleNotFoundError:
            pass
    from tools.odin.valhalla.dashboard.tabs import _placeholder

    return _placeholder.render(dispatch_id, tab_id)
