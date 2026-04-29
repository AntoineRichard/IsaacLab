# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static layout for /<id>/task-drilldown."""

from __future__ import annotations

from dash import dcc, html

__all__ = ["build_layout"]


def build_layout(dispatch_id: str) -> html.Div:
    """Return Tab B's static layout.

    Stores carry per-page state (URL selection, trend metric, trend mode).
    Slots are empty Divs that callbacks populate after the URL has been
    parsed and the picker initialised.
    """
    return html.Div(
        id="tab-b-root",
        children=[
            dcc.Store(id="tab-b-dispatch-id", storage_type="memory", data=dispatch_id),
            dcc.Store(id="tab-b-selection", storage_type="memory", data=None),
            dcc.Store(id="tab-b-trend-metric", storage_type="memory", data="reward_final_ema"),
            dcc.Store(id="tab-b-trend-mode", storage_type="memory", data="ribbon"),
            html.Div(id="tab-b-picker"),
            html.Div(id="tab-b-curves"),
            html.Div(id="tab-b-stats"),
            html.Div(id="tab-b-trend"),
        ],
    )
