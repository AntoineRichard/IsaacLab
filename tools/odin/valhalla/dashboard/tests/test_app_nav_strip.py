# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the persistent top nav strip."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.app import compute_nav_strip


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None or isinstance(child, str):
                continue
            yield from _walk(child)
    elif not isinstance(children, str):
        yield from _walk(children)


def _pills(component):
    """Return components whose className contains 'odin-nav-pill'."""
    out = []
    for c in _walk(component):
        cls = getattr(c, "className", "") or ""
        if "odin-nav-pill" in cls.split():
            out.append(c)
    return out


def test_nav_strip_renders_three_pills():
    out = compute_nav_strip("/")
    assert len(_pills(out)) == 3


def test_nav_strip_landing_marks_dispatches_active():
    out = compute_nav_strip("/")
    pills = _pills(out)
    labels = [getattr(p, "children", None) for p in pills]
    assert labels == ["Dispatches", "Fleet & Jobs", "Task Deep-dive"]
    assert "active" in (getattr(pills[0], "className", "") or "").split()
    assert "active" not in (getattr(pills[1], "className", "") or "").split()
    assert "active" not in (getattr(pills[2], "className", "") or "").split()


def test_nav_strip_landing_disables_dispatch_scoped_pills():
    out = compute_nav_strip("/")
    pills = _pills(out)
    # On the landing page, no dispatch is selected so Fleet/Task pills are disabled.
    assert "disabled" in (getattr(pills[1], "className", "") or "").split()
    assert "disabled" in (getattr(pills[2], "className", "") or "").split()
    # The Dispatches pill is never disabled.
    assert "disabled" not in (getattr(pills[0], "className", "") or "").split()


def test_nav_strip_dispatch_fleet_marks_fleet_active():
    out = compute_nav_strip("/20260427-141302/dispatch-fleet")
    pills = _pills(out)
    classes = [(getattr(p, "className", "") or "").split() for p in pills]
    assert "active" not in classes[0]
    assert "active" in classes[1]
    assert "active" not in classes[2]
    # All three are enabled when a dispatch is in scope.
    for cls in classes:
        assert "disabled" not in cls


def test_nav_strip_task_drilldown_marks_task_active():
    out = compute_nav_strip("/20260427-141302/task-drilldown")
    pills = _pills(out)
    classes = [(getattr(p, "className", "") or "").split() for p in pills]
    assert "active" not in classes[0]
    assert "active" not in classes[1]
    assert "active" in classes[2]


def test_nav_strip_dispatch_scoped_hrefs_carry_dispatch_id():
    out = compute_nav_strip("/20260427-141302/dispatch-fleet")
    pills = _pills(out)
    # Pills are dcc.Link in the enabled state.
    fleet_href = getattr(pills[1], "href", None)
    task_href = getattr(pills[2], "href", None)
    assert fleet_href == "/20260427-141302/dispatch-fleet"
    assert task_href == "/20260427-141302/task-drilldown"


def test_nav_strip_dispatch_scoped_hrefs_preserve_query_string():
    """Switching from Tab A → Tab B should keep the row selection deep-link
    intact when the user already has one set on Tab B."""
    out = compute_nav_strip(
        "/20260427-141302/task-drilldown",
        search="?task=Isaac-Ant-Direct-v0&framework=rsl_rl&backend=physx",
    )
    pills = _pills(out)
    task_href = getattr(pills[2], "href", None)
    assert task_href == ("/20260427-141302/task-drilldown?task=Isaac-Ant-Direct-v0&framework=rsl_rl&backend=physx")


def test_nav_strip_unknown_path_falls_back_to_landing_active():
    out = compute_nav_strip("/garbage/xyz")
    pills = _pills(out)
    classes = [(getattr(p, "className", "") or "").split() for p in pills]
    # No dispatch in URL → Dispatches active, others disabled.
    assert "active" in classes[0]
    assert "disabled" in classes[1]
    assert "disabled" in classes[2]
