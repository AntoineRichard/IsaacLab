# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A jobs-table render tests for the cancel (kill / skip) button."""

from __future__ import annotations

from dash import html

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import (
    render_jobs_section,
)


def _payload_with_jobs(jobs, *, ended_at=None):
    return {
        "dispatch_id": "20260504-100000",
        "ended_at": ended_at,
        "jobs": jobs,
    }


def _job(status: str, run_id: str = "r1") -> dict:
    return {
        "run_id": run_id,
        "task_id": "t",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
    }


def _walk(node):
    """Iterate every Dash component in a tree."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _find_buttons(section, run_id: str) -> list[html.Button]:
    out = []
    for node in _walk(section):
        if not isinstance(node, html.Button):
            continue
        ident = getattr(node, "id", None)
        if isinstance(ident, dict) and ident.get("type") == "tab-a-cancel-toggle" and ident.get("run_id") == run_id:
            out.append(node)
    return out


def test_pending_row_renders_skip_button():
    section = render_jobs_section(_payload_with_jobs([_job("pending", "r1")]))

    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    assert "Skip" in (buttons[0].children or "")


def test_running_row_renders_kill_button():
    """Running rows get the destructive-action ✕ glyph (with the
    ``tab-a-cancel-toggle-kill`` red modifier class) — high-contrast
    cue that the action stops a live trainer. The class is the
    contract; the glyph is rendered text."""
    section = render_jobs_section(_payload_with_jobs([_job("running", "r1")]))

    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    btn = buttons[0]
    assert btn.children == "✕"
    classes = (btn.className or "").split()
    assert "tab-a-cancel-toggle-kill" in classes


def test_completed_row_does_not_render_cancel_button():
    section = render_jobs_section(_payload_with_jobs([_job("completed", "r1")]))

    assert _find_buttons(section, "r1") == []


def test_finished_dispatch_hides_cancel_button():
    section = render_jobs_section(
        _payload_with_jobs([_job("running", "r1")], ended_at="2026-05-04T11:00:00Z"),
    )

    assert _find_buttons(section, "r1") == []


def test_pending_cancellation_renders_pending_badge():
    section = render_jobs_section(
        _payload_with_jobs([_job("running", "r1")]),
        cancel_queue={"r1": "kill"},
    )

    # Button still rendered but in "pending" disabled style.
    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    assert "kill pending" in (buttons[0].title or "").lower()
    # Badge text appears in the Status cell.
    found_badge = any(getattr(node, "className", "") == "tab-a-cancel-pending-badge" for node in _walk(section))
    assert found_badge


def test_confirm_state_renders_red_confirm_label():
    """When run_id is in cancel_confirm (first click within window), the button
    label flips to 'Confirm Kill' / 'Confirm Skip' with the red CSS class."""
    section = render_jobs_section(
        _payload_with_jobs([_job("running", "r1")]),
        cancel_confirm={"r1": 1_700_000_005_000},  # any future ms
    )

    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    label = buttons[0].children or ""
    assert "Confirm Kill" in label
    assert "tab-a-cancel-toggle-confirm" in (buttons[0].className or "")


def test_confirm_state_for_pending_status_says_confirm_skip():
    section = render_jobs_section(
        _payload_with_jobs([_job("pending", "r1")]),
        cancel_confirm={"r1": 1_700_000_005_000},
    )

    buttons = _find_buttons(section, "r1")
    assert "Confirm Skip" in (buttons[0].children or "")


def test_failed_row_does_not_render_cancel_button():
    """Failed terminal status (most common terminal kind in practice) → no
    cancel button. Companion to test_completed_row_does_not_render_cancel_button.
    """
    section = render_jobs_section(_payload_with_jobs([_job("failed", "r1")]))

    assert _find_buttons(section, "r1") == []
