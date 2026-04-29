# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Tab B's stats panel: aggregate card + per-seed table."""

from __future__ import annotations

from dash import html

__all__ = ["render_aggregate_card", "render_seeds_table", "render_stats_panel"]


_METRIC_LABELS = [
    ("reward_final_ema", "Reward", ""),
    ("ep_length_final_ema", "Ep length", ""),
    ("iter_time_s_mean", "Iter time", " s"),
    ("env_steps_per_s_mean", "env_steps/s", ""),
    ("ram_gb_peak", "RAM peak", " GB"),
    ("gpu_mem_gb_peak", "GPU mem peak", " GB"),
]


def render_aggregate_card(aggregate_block: dict, divergent_seeds: list[str]) -> html.Div:
    """Build the left-side aggregate card.

    aggregate_block: dict matching aggregate.json.rows[].aggregate.
    Renders one stat-line per metric ('label: mean ± std cv X.X%') plus a
    closing divergent-seeds line.
    """
    lines: list = []
    for key, label, unit in _METRIC_LABELS:
        block = aggregate_block.get(key)
        if not isinstance(block, dict):
            continue
        mean = float(block.get("mean", 0.0))
        std = float(block.get("std", 0.0))
        cv = float(block.get("cv_pct", 0.0))
        cv_class = _cv_class(cv)
        lines.append(
            html.Div(
                className="tab-b-stat-line",
                children=[
                    html.Span(label, className="tab-b-stat-label"),
                    html.Span(
                        children=[
                            html.Strong(_fmt_num(mean) + unit),
                            html.Span(f"  ± {_fmt_num(std)}{unit}"),
                            html.Span(f"  cv {cv:.1f}%", className=f"tab-b-cv {cv_class}"),
                        ],
                        className="tab-b-stat-value",
                    ),
                ],
            )
        )
    div_text = "—" if not divergent_seeds else ", ".join(f"seed {s}" for s in divergent_seeds)
    lines.append(
        html.Div(
            className="tab-b-stat-line tab-b-divergent-line",
            children=[
                html.Span("Divergent seeds", className="tab-b-stat-label"),
                html.Span(div_text, className="tab-b-stat-value"),
            ],
        )
    )
    return html.Div(
        id="tab-b-aggregate-card",
        className="tab-b-stats-card",
        children=[
            html.Div("Aggregate", className="tab-b-stats-card-title"),
            *lines,
        ],
    )


def render_seeds_table(seeds_block: dict) -> html.Div:
    """Build the right-side per-seed table (11 columns)."""
    headers = [
        "Seed",
        "Status",
        "Reward",
        "Ep length",
        "Iter time",
        "env_steps/s",
        "RAM peak",
        "GPU mem",
        "Wall time",
        "Startup",
        "Host",
    ]
    header_row = html.Tr(children=[html.Th(h) for h in headers])
    body_rows: list = []
    for seed_key in sorted(seeds_block.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
        seed = seeds_block[seed_key]
        body_rows.append(_seed_row(seed_key, seed))
    return html.Div(
        id="tab-b-seeds-table",
        className="tab-b-seeds-table-wrapper",
        children=[
            html.Table(
                children=[html.Thead(children=[header_row]), html.Tbody(children=body_rows)],
                className="tab-b-seeds-table",
            )
        ],
    )


def render_stats_panel(aggregate_payload_row: dict) -> html.Div:
    """Combine aggregate card + per-seed table in a two-column row."""
    aggregate_block = aggregate_payload_row.get("aggregate", {}) or {}
    seeds_block = aggregate_payload_row.get("seeds", {}) or {}
    divergent = aggregate_payload_row.get("divergent_seeds", []) or []
    return html.Div(
        id="tab-b-stats-content",
        className="tab-b-stats-row",
        children=[
            render_aggregate_card(aggregate_block, divergent),
            render_seeds_table(seeds_block),
        ],
    )


def _seed_row(seed_key: str, seed: dict) -> html.Tr:
    status = str(seed.get("status", "unknown"))
    pill_label = {"completed": "OK", "failed": "Failed"}.get(status, status.capitalize())
    pill = html.Span(
        pill_label,
        className=f"tab-b-pill tab-b-seed-status-{status}",
    )
    return html.Tr(
        children=[
            html.Td(seed_key, className="tab-b-seed-id"),
            html.Td(pill),
            html.Td(_fmt_or_dash(seed.get("reward_final_ema")), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("ep_length_final_ema")), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("iter_time_s_mean"), suffix=" s"), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("env_steps_per_s_mean"), int_fmt=True), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("ram_gb_peak"), suffix=" GB"), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("gpu_mem_gb_peak"), suffix=" GB"), className="tab-b-mono"),
            html.Td(_fmt_wall_time(seed.get("total_wall_time_s")), className="tab-b-mono"),
            html.Td(_fmt_startup_phases(seed), className="tab-b-mono"),
            html.Td(str(seed.get("assigned_to") or "—"), className="tab-b-mono tab-b-muted"),
        ]
    )


def _cv_class(cv_pct: float) -> str:
    if cv_pct <= 5.0:
        return "tab-b-cv-good"
    if cv_pct <= 15.0:
        return "tab-b-cv-warn"
    return "tab-b-cv-bad"


def _fmt_num(value) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_or_dash(value, *, suffix: str = "", int_fmt: bool = False) -> str:
    if value is None:
        return "—"
    if int_fmt:
        return f"{int(value)}{suffix}"
    return f"{_fmt_num(float(value))}{suffix}"


def _fmt_wall_time(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = int(float(seconds))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes}:{sec:02d}"


def _fmt_startup_phases(seed: dict) -> str:
    parts = [seed.get("startup_app_launch_s"), seed.get("startup_env_creation_s"), seed.get("startup_first_step_s")]
    if all(p is None for p in parts):
        return "—"
    rendered = []
    for p in parts:
        if p is None:
            rendered.append("—")
        elif p < 1:
            rendered.append(f"{p:.3f}")
        else:
            rendered.append(f"{p:.1f}")
    return " / ".join(rendered)
