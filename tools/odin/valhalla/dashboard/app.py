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

__all__ = ["compute_nav_strip", "create_app", "route_pathname"]


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
    """Build the SPA shell.

    When ``initial_dispatch`` is set (CLI invoked with a dispatch_id),
    we explicitly push the URL to ``/<id>/dispatch-fleet`` so the page
    opens directly on Tab A even if the user typed just ``/`` in the
    browser.  When ``initial_dispatch`` is None, we leave the
    ``pathname`` unset so ``dcc.Location`` picks up whatever URL the
    browser is on — otherwise we'd force every URL back to ``/`` and
    only the landing page would ever render.
    """
    location_kwargs: dict = {"id": "url", "refresh": False}
    if initial_dispatch is not None:
        location_kwargs["pathname"] = f"/{initial_dispatch.name}/dispatch-fleet"
    return html.Div(
        id="app-root",
        children=[
            dcc.Location(**location_kwargs),
            dcc.Store(id="active-dispatch", storage_type="memory"),
            _build_banner(),
            html.Div(id="odin-nav-strip"),
            html.Div(id="page-content"),
        ],
    )


def _build_banner() -> html.Div:
    """Top branding banner — NVIDIA-green stripe + ODIN wordmark + tagline.

    Renders on every page (landing + per-dispatch tabs). The NVIDIA logo
    asset lives at ``assets/nvidia-logo.svg`` and is auto-served by Dash.
    """
    return html.Div(
        id="odin-banner",
        children=[
            html.Div(className="odin-banner-stripe"),
            html.Div(
                className="odin-banner-row",
                children=[
                    html.Img(src="/assets/nvidia-logo.svg", className="odin-banner-nvidia-logo", alt="NVIDIA"),
                    html.Div(
                        className="odin-banner-titles",
                        children=[
                            html.Div("ODIN", className="odin-banner-name"),
                            html.Div(
                                "IsaacLab internal training evaluation harness",
                                className="odin-banner-tagline",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


_NAV_TABS: list[tuple[str, str]] = [
    ("dispatches", "Dispatches"),
    ("dispatch-fleet", "Fleet & Jobs"),
    ("task-drilldown", "Task Drill-down"),
]


def compute_nav_strip(pathname: str, *, search: str = "") -> html.Div:
    """Build the persistent top nav strip for the given URL.

    The strip has three pills (Dispatches / Fleet & Jobs / Task Drill-down).
    Exactly one is marked ``active``; the dispatch-scoped pills are marked
    ``disabled`` (rendered as ``html.Span``) when no dispatch is in scope.
    Otherwise dispatch-scoped pills are rendered as ``dcc.Link`` and carry
    the dispatch_id forward in their ``href``. ``search`` is preserved on
    the active dispatch-scoped pill so query-string deep-links survive
    when the user returns to the same tab.
    """
    parts = [p for p in pathname.split("/") if p]
    dispatch_id: str | None = None
    active_tab: str = "dispatches"
    if parts and _DISPATCH_ID_RE.match(parts[0]):
        dispatch_id = parts[0]
        if len(parts) >= 2 and parts[1] in {"dispatch-fleet", "task-drilldown"}:
            active_tab = parts[1]

    children = []
    for slug, label in _NAV_TABS:
        children.append(_nav_pill(slug, label, dispatch_id, active_tab, search))
    return html.Div(className="odin-nav-strip", children=children)


def _nav_pill(
    slug: str,
    label: str,
    dispatch_id: str | None,
    active_tab: str,
    search: str,
) -> html.Span | dcc.Link:
    is_active = slug == active_tab
    classes = ["odin-nav-pill"]
    if is_active:
        classes.append("active")

    if slug == "dispatches":
        return dcc.Link(label, href="/", className=" ".join(classes))

    # Dispatch-scoped pill — disabled when no dispatch is in URL scope.
    if dispatch_id is None:
        classes.append("disabled")
        return html.Span(
            label,
            className=" ".join(classes),
            title="Pick a dispatch first",
        )
    href = f"/{dispatch_id}/{slug}"
    if is_active and search:
        href = f"{href}{search}"
    return dcc.Link(label, href=href, className=" ".join(classes))


def _register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def _on_url(pathname: str):
        return route_pathname(pathname or "/", data)

    @app.callback(
        Output("odin-nav-strip", "children"),
        Input("url", "pathname"),
        Input("url", "search"),
    )
    def _on_nav_url(pathname: str, search: str):
        return compute_nav_strip(pathname or "/", search=search or "")

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
