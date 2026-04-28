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

__all__ = ["build_layout"]


_TICK_MS = 5_000


def build_layout(dispatch_id: str) -> html.Div:
    """Return the Tab A static layout for ``dispatch_id``.

    Stores carry per-page state (filters, expansion set, ssh-tail cache).
    Slots are empty Divs that callbacks populate on every tick.
    """
    return html.Div(
        id="tab-a-root",
        children=[
            dcc.Interval(id="tab-a-tick", interval=_TICK_MS, n_intervals=0),
            dcc.Store(id="tab-a-dispatch-id", storage_type="memory", data=dispatch_id),
            dcc.Store(id="tab-a-failure-filter", storage_type="memory", data=None),
            dcc.Store(id="tab-a-expanded-run-ids", storage_type="memory", data=[]),
            dcc.Store(id="tab-a-ssh-tail-store", storage_type="memory", data={}),
            html.Div(id="tab-a-header"),
            html.Div(id="tab-a-fleet-table"),
            html.Div(id="tab-a-jobs-section"),
        ],
    )
