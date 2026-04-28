# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A header: title, live/done pill, totals, failure pills."""

from __future__ import annotations

from collections import Counter

from dash import html

__all__ = ["render_header"]


def render_header(dispatch_payload: dict) -> html.Div:
    """Build the header strip for a parsed dispatch.json payload.

    Pure render — no Dash callbacks, no DataLayer access. Returns a Div with
    id ``tab-a-header-content`` containing the title row, live/done pill,
    totals row, and (when there are failures) the click-to-filter pill row.
    """
    dispatch_id = str(dispatch_payload.get("dispatch_id", ""))
    commit_sha = str(dispatch_payload.get("commit_sha", "") or "")
    ended_at = dispatch_payload.get("ended_at")
    fleet = dispatch_payload.get("fleet", []) or []
    jobs = dispatch_payload.get("jobs", []) or []
    skipped = dispatch_payload.get("skipped", []) or []

    title_parts = [f"Dispatch {dispatch_id}"]
    if commit_sha:
        title_parts.append(f"commit {commit_sha[:7]}")
    title_parts.append(f"{len(fleet)} hosts")
    title_text = " · ".join(title_parts)

    pill = (
        html.Span("● Live", className="tab-a-live-pill")
        if ended_at is None
        else html.Span("✓ Done", className="tab-a-done-pill")
    )

    by_status = Counter(j.get("status", "unknown") for j in jobs)
    totals_text = (
        f"{len(jobs)} total · "
        f"{by_status.get('completed', 0)} completed · "
        f"{by_status.get('failed', 0)} failed · "
        f"{by_status.get('pending', 0)} pending · "
        f"{len(skipped)} skipped"
    )

    children: list = [
        html.Div(title_text, className="tab-a-header-title"),
        html.Div(pill, className="tab-a-header-pill-row"),
        html.Div(totals_text, className="tab-a-header-totals"),
    ]

    by_kind = Counter(
        (j.get("failure") or {}).get("kind", "unknown")
        for j in jobs
        if j.get("status") == "failed" and j.get("failure")
    )
    if by_kind:
        pill_children: list = [html.Span(f"Failures: {sum(by_kind.values())}  ")]
        for kind, count in sorted(by_kind.items()):
            pill_children.append(
                html.Button(
                    f"{kind}: {count}",
                    id={"type": "tab-a-failure-pill", "kind": kind},
                    n_clicks=0,
                    className=f"tab-a-failure-pill tab-a-failure-pill-{kind}",
                )
            )
        children.append(html.Div(pill_children, className="tab-a-header-failure-pills"))

    return html.Div(id="tab-a-header-content", children=children)
