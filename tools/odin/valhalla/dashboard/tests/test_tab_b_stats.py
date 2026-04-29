# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.stats."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.stats import (
    render_aggregate_card,
    render_seeds_table,
    render_stats_panel,
)


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None or isinstance(child, str):
                continue
            yield from _walk(child)
    elif not isinstance(children, str):
        yield from _walk(children)


def _text_blob(component) -> str:
    parts: list[str] = []
    for c in _walk(component):
        ch = getattr(c, "children", None)
        if isinstance(ch, str):
            parts.append(ch)
    return " ".join(parts)


def _has_class(component, cls: str) -> bool:
    for c in _walk(component):
        c_cls = getattr(c, "className", "") or ""
        if cls in c_cls.split():
            return True
    return False


def _has_id(component, target_id: str) -> bool:
    return any(getattr(c, "id", None) == target_id for c in _walk(component))


def _aggregate_block(cv_pct: float = 3.2) -> dict:
    return {
        "n_seeds_completed": 3,
        "n_seeds_failed": 0,
        "reward_final_ema": {"mean": 7991.51, "std": 257.68, "min": 7795.39, "max": 8283.34, "cv_pct": cv_pct},
        "ep_length_final_ema": {"mean": 839.76, "std": 16.36, "min": 822.41, "max": 854.91, "cv_pct": 1.95},
        "iter_time_s_mean": {"mean": 1.828, "std": 0.033, "min": 1.803, "max": 1.865, "cv_pct": 1.78},
        "env_steps_per_s_mean": {"mean": 72670.28, "std": 1680.0, "min": 70762.97, "max": 73930.63, "cv_pct": 2.31},
        "ram_gb_peak": {"mean": 4.57, "std": 0.02, "min": 4.55, "max": 4.58, "cv_pct": 0.33},
        "gpu_mem_gb_peak": {"mean": 4.24, "std": 0.0, "min": 4.24, "max": 4.24, "cv_pct": 0.0},
    }


def _seed(*, status: str = "completed", **overrides) -> dict:
    base = {
        "run_id": "rsl-rl_physx_X_seed42",
        "status": status,
        "assigned_to": "10.0.0.1",
        "reward_final_ema": 7795.39,
        "ep_length_final_ema": 822.41,
        "iter_time_s_mean": 1.815,
        "iter_time_s_std": 0.217,
        "env_steps_per_s_mean": 73317.24,
        "iterations_completed": 1000,
        "total_wall_time_s": 1815.11,
        "ram_gb_peak": 4.57,
        "gpu_mem_gb_peak": 4.24,
        "startup_app_launch_s": 3.52,
        "startup_env_creation_s": 13.87,
        "startup_first_step_s": 0.002,
    }
    base.update(overrides)
    return base


def test_aggregate_card_renders_one_line_per_metric():
    component = render_aggregate_card(_aggregate_block(), divergent_seeds=[])
    text = _text_blob(component)
    for label in ("Reward", "Ep length", "Iter time", "env_steps/s", "RAM peak", "GPU mem peak"):
        assert label in text, f"missing label {label!r}"


def test_aggregate_card_formats_mean_pm_std():
    text = _text_blob(render_aggregate_card(_aggregate_block(), divergent_seeds=[]))
    assert "7991.51" in text
    assert "257.68" in text
    assert "cv 3.2%" in text


def test_aggregate_card_cv_color_green_below_5pct():
    component = render_aggregate_card(_aggregate_block(cv_pct=3.2), divergent_seeds=[])
    assert _has_class(component, "tab-b-cv-good")


def test_aggregate_card_cv_color_orange_5_to_15pct():
    component = render_aggregate_card(_aggregate_block(cv_pct=8.0), divergent_seeds=[])
    assert _has_class(component, "tab-b-cv-warn")


def test_aggregate_card_cv_color_red_above_15pct():
    component = render_aggregate_card(_aggregate_block(cv_pct=20.0), divergent_seeds=[])
    assert _has_class(component, "tab-b-cv-bad")


def test_aggregate_card_lists_divergent_seeds():
    component = render_aggregate_card(_aggregate_block(), divergent_seeds=["43"])
    assert "seed 43" in _text_blob(component)


def test_aggregate_card_no_divergent_seeds_renders_dash():
    component = render_aggregate_card(_aggregate_block(), divergent_seeds=[])
    assert "—" in _text_blob(component)


def test_seeds_table_one_row_per_seed():
    seeds = {"42": _seed(), "43": _seed(), "44": _seed()}
    component = render_seeds_table(seeds)
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    assert len(rows) == 4  # 1 header + 3 seeds


def test_seeds_table_column_set():
    seeds = {"42": _seed()}
    component = render_seeds_table(seeds)
    headers = [c for c in _walk(component) if type(c).__name__ == "Th"]
    expected = [
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
    actual = [getattr(h, "children", "") for h in headers]
    assert actual == expected


def test_seeds_table_status_pill_for_completed():
    seeds = {"42": _seed(status="completed")}
    component = render_seeds_table(seeds)
    assert _has_class(component, "tab-b-seed-status-completed")


def test_seeds_table_status_pill_for_failed():
    seeds = {"42": _seed(status="failed")}
    component = render_seeds_table(seeds)
    assert _has_class(component, "tab-b-seed-status-failed")


def test_seeds_table_dashes_when_metric_missing():
    seed = {"run_id": "x", "status": "failed", "assigned_to": "10.0.0.1"}
    component = render_seeds_table({"42": seed})
    text = _text_blob(component)
    # Many dashes because no metrics present.
    assert text.count("—") >= 6


def test_render_stats_panel_contains_both_cards():
    row = {
        "task": "X",
        "framework": "rsl_rl",
        "backend": "physx",
        "aggregate": _aggregate_block(),
        "seeds": {"42": _seed()},
        "divergent_seeds": [],
    }
    component = render_stats_panel(row)
    assert _has_id(component, "tab-b-aggregate-card")
    assert _has_id(component, "tab-b-seeds-table")
