# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static layout for /<id>/dispatch-fleet.

All dynamic content lives in slots (id="tab-a-..."); callbacks fill them on
mount and on every dcc.Interval tick.
"""

from __future__ import annotations

from dash import dcc, html

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import render_filter_row

__all__ = ["build_layout"]


_TICK_MS = 5_000


def build_layout(dispatch_id: str) -> html.Div:
    """Return the Tab A static layout for ``dispatch_id``.

    Stores carry per-page state (filters, expansion set, ssh-tail cache).
    The filter row is rendered statically so its component IDs exist at
    cold mount — without that, the update_jobs callback's Inputs (which
    reference the dropdown / input values) would point to nonexistent
    components, and Dash with ``suppress_callback_exceptions=True`` would
    silently never fire the callback.  All other dynamic content lives in
    empty slots populated by callbacks on every tick.
    """
    return html.Div(
        id="tab-a-root",
        children=[
            dcc.Interval(id="tab-a-tick", interval=_TICK_MS, n_intervals=0),
            dcc.Store(id="tab-a-dispatch-id", storage_type="memory", data=dispatch_id),
            dcc.Store(id="tab-a-failure-filter", storage_type="memory", data=None),
            dcc.Store(id="tab-a-expanded-run-ids", storage_type="memory", data=[]),
            dcc.Store(id="tab-a-ssh-tail-store", storage_type="memory", data={}),
            # Bumped on every retry-toggle click so the jobs-poll callback
            # re-fires immediately and the row's button + banner refresh
            # without waiting for the 5 s dcc.Interval.
            dcc.Store(id="tab-a-retry-bump", storage_type="memory", data=0),
            html.Div(id="tab-a-header"),
            html.Div(id="tab-a-fleet-table"),
            html.Div(
                id="tab-a-jobs-section",
                children=[
                    render_filter_row(),
                    html.Div(id="tab-a-jobs-rows-content"),
                ],
            ),
        ],
    )
