# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Tab B's metric trend section: selector + line/bar chart over N dispatches."""

from __future__ import annotations

import contextlib
import statistics
import sys

import plotly.graph_objects as go
from dash import dcc

__all__ = [
    "_TREND_METRICS",
    "compute_trend_points",
    "render_trend_chart",
]


_TREND_METRICS = [
    {"value": "reward_final_ema", "label": "Reward (final EMA)", "source": "aggregate"},
    {"value": "ep_length_final_ema", "label": "Episode length (final EMA)", "source": "aggregate"},
    {"value": "iter_time_s_mean", "label": "Iter time", "source": "aggregate"},
    {"value": "env_steps_per_s_mean", "label": "Env steps / s", "source": "aggregate"},
    {"value": "ram_gb_peak", "label": "RAM peak", "source": "aggregate"},
    {"value": "gpu_mem_gb_peak", "label": "GPU mem peak", "source": "aggregate"},
    {"value": "startup_app_launch_s", "label": "Startup: app launch", "source": "seeds"},
    {"value": "startup_env_creation_s", "label": "Startup: env creation", "source": "seeds"},
    {"value": "startup_first_step_s", "label": "Startup: first step", "source": "seeds"},
]
_METRIC_SOURCES = {m["value"]: m["source"] for m in _TREND_METRICS}


def compute_trend_points(
    data,
    dispatch_ids: list[str],
    task: str,
    framework: str,
    backend: str,
    metric: str,
) -> list[dict]:
    """For each dispatch_id, compute one point for the metric.

    Each point: {dispatch_id, commit_sha, mean, std, n_seeds_completed}.

    Skips dispatches whose aggregate.json is missing or whose rows[] don't
    include the requested (task, framework, backend) tuple. Logs a [WARNING]
    for any dispatch whose aggregate read raises.
    """
    source = _METRIC_SOURCES.get(metric, "aggregate")
    out: list[dict] = []
    for did in dispatch_ids:
        try:
            agg = data.load_aggregate(did)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b trend: load_aggregate({did}) raised: {exc}", file=sys.stderr)
            continue
        if agg is None:
            continue
        row = _find_row(agg, task, framework, backend)
        if row is None:
            continue
        try:
            mean, std = _metric_value(row, metric, source)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b trend: metric extract failed for {did}: {exc}", file=sys.stderr)
            continue
        commit_sha = ""
        with contextlib.suppress(Exception):
            commit_sha = str(data.load_dispatch(did).get("commit_sha", "") or "")
        n_seeds = (row.get("aggregate") or {}).get("n_seeds_completed", 0)
        out.append(
            {
                "dispatch_id": did,
                "commit_sha": commit_sha,
                "mean": mean,
                "std": std,
                "n_seeds_completed": int(n_seeds or 0),
            }
        )
    return out


def render_trend_chart(points: list[dict], metric_label: str, *, mode: str = "ribbon") -> dcc.Graph:
    """Render the trend chart.

    mode='ribbon' → mean line + fill band of (mean ± std).
    mode='bars'   → bar chart with vertical error_y whiskers.
    """
    n = len(points)
    if n == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", height=240)
        return dcc.Graph(id="tab-b-trend-chart", figure=empty_fig, config={"displayModeBar": False})

    xs = list(range(n))
    means = [p["mean"] for p in points]
    stds = [p["std"] for p in points]
    short_shas = [(p["commit_sha"][:7] or "—") for p in points]
    labels = [f"{p['commit_sha'][:7] or '—'}<br>n={p['n_seeds_completed']}<br>{p['dispatch_id']}" for p in points]

    # Mark the rightmost (newest) point as "current".
    short_shas[-1] = short_shas[-1] + " ▲"

    fig = go.Figure()
    if mode == "bars":
        marker_colors = ["#66b6ff"] * (n - 1) + ["#76b900"]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=means,
                error_y={"type": "data", "array": stds, "color": "#aaa"},
                marker={"color": marker_colors},
                hovertext=labels,
                hovertemplate="%{hovertext}<br>" + metric_label + ": %{y:.2f}<extra></extra>",
                name=metric_label,
            )
        )
    else:
        # Ribbon: fill band (upper-then-lower) drawn first, then mean line on top.
        upper = [m + s for m, s in zip(means, stds)]
        lower = [m - s for m, s in zip(means, stds)]
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(102,182,255,0.18)",
                line={"color": "rgba(0,0,0,0)"},
                name="±std",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        marker_colors = ["#66b6ff"] * (n - 1) + ["#76b900"]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line={"color": "#66b6ff", "width": 2},
                marker={"color": marker_colors, "size": 8, "line": {"color": "#fff", "width": 1}},
                hovertext=labels,
                hovertemplate="%{hovertext}<br>" + metric_label + ": %{y:.2f}<extra></extra>",
                name=metric_label,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        margin={"l": 60, "r": 20, "t": 30, "b": 60},
        xaxis={
            "tickmode": "array",
            "tickvals": xs,
            "ticktext": short_shas,
        },
        yaxis_title=metric_label,
        height=260,
        showlegend=False,
    )
    return dcc.Graph(id="tab-b-trend-chart", figure=fig, config={"displayModeBar": False})


def _find_row(aggregate: dict, task: str, framework: str, backend: str) -> dict | None:
    for row in aggregate.get("rows", []) or []:
        if row.get("task") == task and row.get("framework") == framework and row.get("backend") == backend:
            return row
    return None


def _metric_value(row: dict, metric: str, source: str) -> tuple[float, float]:
    if source == "aggregate":
        block = (row.get("aggregate") or {}).get(metric)
        if not isinstance(block, dict):
            raise KeyError(metric)
        return float(block.get("mean", 0.0)), float(block.get("std", 0.0))
    seeds = row.get("seeds") or {}
    values = []
    for seed in seeds.values():
        v = seed.get(metric)
        if v is not None:
            values.append(float(v))
    if not values:
        raise ValueError(f"no per-seed values for {metric}")
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std
