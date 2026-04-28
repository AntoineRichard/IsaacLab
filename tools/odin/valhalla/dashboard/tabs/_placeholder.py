# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spec-0 placeholder rendered for any tab whose real implementation hasn't landed yet."""

from __future__ import annotations

from dash import html


def render(dispatch_id: str, tab_id: str):
    """Return the Spec-0 placeholder component for ``<dispatch_id>/<tab_id>``."""
    spec_number = {"dispatch-fleet": 1, "task-drilldown": 2, "startup": 3}.get(tab_id, "?")
    return html.Div(
        id="tab-placeholder",
        children=[
            html.H3(f"Tab '{tab_id}'"),
            html.P(
                f"Coming in Spec {spec_number}. Dashboard skeleton (Spec 0) ships only "
                f"the multi-dispatch routing and dispatch picker.",
            ),
            html.P(f"Active dispatch: {dispatch_id}"),
        ],
    )
