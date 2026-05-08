# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heimdall card — sibling panel of ``fleet_table.py`` on the dispatch-fleet tab.

Renders the most recent :func:`~tools.odin.asgard.heimdall.read_fleet_json`
payload as a small card showing the watcher's last look-up timestamp,
per-host status row, and a tail of recent activity events.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dash import html

__all__ = ["render_empty_state", "render_heimdall_card"]


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_str(generated_at: str, now_iso: str | None) -> str:
    g = _parse_iso(generated_at)
    if g is None:
        return "unknown"
    if now_iso is None:
        now = datetime.now(timezone.utc)
    else:
        now = _parse_iso(now_iso) or datetime.now(timezone.utc)
    delta_s = max(0, int((now - g).total_seconds()))
    if delta_s < 60:
        return f"{delta_s}s ago"
    minutes, secs = divmod(delta_s, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s ago"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m ago"


def _host_row(name: str, h: dict) -> html.Div:
    if h.get("quarantined"):
        icon, status = "⛔", f"quarantined ({h.get('failure_reason') or 'unknown'})"
    elif h.get("healthy"):
        icon, status = "✓", "healthy"
    else:
        icon, status = "✗", f"unhealthy ({h.get('failure_reason') or 'unknown'})"
    return html.Div(
        children=[
            html.Span(icon, className="heimdall-icon"),
            html.Span(f" {name}: ", className="heimdall-host"),
            html.Span(status, className="heimdall-status"),
        ],
        className="heimdall-host-row",
    )


def _event_row(ev: dict) -> html.Div:
    summary = f"{ev.get('ts', '?')} — {ev.get('kind', '?')}"
    host = ev.get("host")
    if host:
        summary += f" {host}"
    reason = ev.get("reason")
    if reason:
        summary += f" ({reason})"
    return html.Div(summary, className="heimdall-event-row")


def render_empty_state() -> html.Div:
    """Placeholder card shown when ``fleet.json`` is missing.

    Older dispatches (pre-Heimdall) don't have one, and a watcher that
    hasn't ticked yet hasn't written one yet either.
    """
    return html.Div(
        children=[
            html.H4("Heimdall"),
            html.Div(
                "Heimdall not active for this dispatch (no fleet.json).",
                className="heimdall-empty",
            ),
        ],
        className="heimdall-card",
    )


def render_heimdall_card(payload: dict, *, now_iso: str | None = None) -> html.Div:
    """Render the panel for one ``fleet.json`` payload.

    Args:
        payload: Raw dict as returned by
            :func:`~tools.odin.asgard.heimdall.read_fleet_json`.
        now_iso: Optional UTC timestamp pinning "X ago" age computation.
            Tests pin this for determinism. Production callers pass
            ``None`` to use wall-clock time.
    """
    if payload is None:
        return render_empty_state()
    generated_at = payload.get("generated_at", "")
    age = _age_str(generated_at, now_iso)
    hosts = payload.get("hosts", {}) or {}
    events = payload.get("recent_events", []) or []
    stale_count = sum(1 for ev in events if ev.get("kind") == "stale_job_killed")

    if events:
        events_section = [
            html.H5("Recent activity"),
            *[_event_row(ev) for ev in events[-5:]],
        ]
    else:
        events_section = [
            html.H5("Recent activity"),
            html.Div("No recent events.", className="heimdall-empty"),
        ]

    return html.Div(
        children=[
            html.H4("Heimdall"),
            html.Div(
                f"Last look-up: {generated_at} ({age})",
                className="heimdall-header",
            ),
            html.Div(
                children=[_host_row(name, h) for name, h in sorted(hosts.items())],
                className="heimdall-hosts",
            ),
            html.Div(children=events_section, className="heimdall-events"),
            html.Div(
                f"Stale jobs killed: {stale_count}" if stale_count else "",
                className="heimdall-stale-badge",
            ),
        ],
        className="heimdall-card",
    )
