# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Dash app factory + routing."""

from __future__ import annotations

import json
from pathlib import Path

import dash

from tools.odin.valhalla.dashboard.app import create_app, route_pathname


def _write_dispatch(runs_root: Path, dispatch_id: str) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "abc",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": [],
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def test_create_app_returns_dash_instance(tmp_path):
    app = create_app(tmp_path)
    assert isinstance(app, dash.Dash)
    assert app.title == "Odin"


def test_create_app_layout_is_non_empty(tmp_path):
    app = create_app(tmp_path)
    assert app.layout is not None


def test_route_pathname_landing(tmp_path):
    """Empty path returns the landing component."""
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/", data)
    assert _has_id(component, "landing-root")


def test_route_pathname_dispatch_redirects_to_tab_a(tmp_path):
    """`/<id>/` returns a redirect (Location component) to /<id>/dispatch-fleet."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/", data)
    assert _has_id(component, "redirect-to-tab-a") or _is_redirect_to(component, "/20260427-141302/dispatch-fleet")


def test_route_pathname_unknown_dispatch_returns_404(tmp_path):
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/does-not-exist/dispatch-fleet", data)
    assert _has_id(component, "not-found-root")


def test_route_pathname_unknown_path_returns_404(tmp_path):
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/garbage/route/here", data)
    assert _has_id(component, "not-found-root")


def test_route_pathname_known_tab_renders_placeholder(tmp_path):
    """Tab path on a real dispatch renders the placeholder (Spec 0 has no tab content)."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer
    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/dispatch-fleet", data)
    assert _has_id(component, "tab-placeholder")


# -- helpers --


def _walk(component):
    """Yield this component plus every descendant in its `children` tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None or isinstance(child, str):
                continue
            yield from _walk(child)
    else:
        if not isinstance(children, str):
            yield from _walk(children)


def _has_id(component, target_id: str) -> bool:
    for c in _walk(component):
        if getattr(c, "id", None) == target_id:
            return True
    return False


def _is_redirect_to(component, expected_href: str) -> bool:
    """A dcc.Location with the expected href is acceptable for redirect."""
    for c in _walk(component):
        if isinstance(c, dash.dcc.Location):
            if getattr(c, "href", None) == expected_href and getattr(c, "refresh", False):
                return True
    return False
