# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Tab B's reward + ep_length overlay charts."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

__all__ = ["render_curves", "_SEED_PALETTE"]


_SEED_PALETTE = ["#66b6ff", "#50c88c", "#ffa500", "#beA0ff", "#e16868"]
_DIVERGENT_COLOR = "#e16868"


def render_curves(
    bundles: dict[str, dict],
    *,
    divergent_seeds: list[str],
) -> html.Div:
    """Build two plotly figures (reward + ep_length) overlaying all seeds.

    Args:
        bundles: ``{seed_str: training_payload}``. Missing series → seed skipped.
        divergent_seeds: seeds drawn with red dashed stroke instead of palette.

    Returns:
        Div(id='tab-b-curves-content') with up to 2 ``dcc.Graph`` components.
    """
    if not bundles:
        return html.Div(
            id="tab-b-curves-content",
            className="tab-b-empty-state",
            children=[html.P("No training.json bundles available for this row.")],
        )

    sorted_seeds = sorted(bundles.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)
    divergent_set = set(str(s) for s in divergent_seeds)

    reward_fig = _build_overlay_figure(sorted_seeds, bundles, "reward", divergent_set, y_label="reward (final EMA)")
    ep_fig = _build_overlay_figure(sorted_seeds, bundles, "ep_length", divergent_set, y_label="ep_length (final EMA)")
    return html.Div(
        id="tab-b-curves-content",
        children=[
            dcc.Graph(id="tab-b-curve-reward", figure=reward_fig, config={"displayModeBar": False}),
            dcc.Graph(id="tab-b-curve-ep-length", figure=ep_fig, config={"displayModeBar": False}),
        ],
    )


def _build_overlay_figure(
    sorted_seeds: list[str],
    bundles: dict[str, dict],
    series_key: str,
    divergent_set: set[str],
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    for idx, seed in enumerate(sorted_seeds):
        learning = bundles[seed].get("learning", {}) or {}
        block = learning.get(series_key) or {}
        series = block.get("series_per_iter")
        if not series:
            continue
        is_divergent = seed in divergent_set
        color = _DIVERGENT_COLOR if is_divergent else _SEED_PALETTE[idx % len(_SEED_PALETTE)]
        dash = "dash" if is_divergent else "solid"
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series))),
                y=series,
                mode="lines",
                name=f"seed {seed}",
                line={"color": color, "dash": dash, "width": 1.6},
                hovertemplate=f"seed {seed}<br>iter %{{x}}<br>{series_key} %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
        xaxis_title="iterations",
        yaxis_title=y_label,
        height=240,
        legend={"orientation": "h", "y": -0.18},
    )
    return fig
