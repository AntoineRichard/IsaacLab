# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.trend."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import (
    compute_trend_points,
    render_trend_chart,
)


def _walk(component):
    """Walk a component tree yielding all descendants."""
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


def _row(
    *,
    task: str = "X",
    framework: str = "rsl_rl",
    backend: str = "physx",
    agg: dict | None = None,
    seeds: dict | None = None,
) -> dict:
    return {
        "task": task,
        "framework": framework,
        "backend": backend,
        "aggregate": agg or {},
        "seeds": seeds or {},
        "divergent_seeds": [],
    }


def _aggregate_block(reward_mean: float = 7991.51, reward_std: float = 257.68) -> dict:
    return {
        "n_seeds_completed": 3,
        "reward_final_ema": {"mean": reward_mean, "std": reward_std, "min": 0.0, "max": 0.0, "cv_pct": 3.2},
        "ep_length_final_ema": {"mean": 839.76, "std": 16.36, "min": 0, "max": 0, "cv_pct": 1.95},
        "iter_time_s_mean": {"mean": 1.828, "std": 0.033, "min": 0, "max": 0, "cv_pct": 1.78},
        "env_steps_per_s_mean": {"mean": 72670.28, "std": 1680.0, "min": 0, "max": 0, "cv_pct": 2.31},
        "ram_gb_peak": {"mean": 4.57, "std": 0.02, "min": 0, "max": 0, "cv_pct": 0.33},
        "gpu_mem_gb_peak": {"mean": 4.24, "std": 0.0, "min": 0, "max": 0, "cv_pct": 0.0},
    }


def _seed(*, startup_app_launch_s: float = 3.5) -> dict:
    return {
        "run_id": "rsl-rl_physx_X_seed42",
        "status": "completed",
        "reward_final_ema": 7795.39,
        "iter_time_s_mean": 1.815,
        "ram_gb_peak": 4.57,
        "startup_app_launch_s": startup_app_launch_s,
        "startup_env_creation_s": 13.87,
        "startup_first_step_s": 0.002,
    }


class _StubData:
    """DataLayer drop-in for trend tests."""

    def __init__(self):
        self._aggregates: dict[str, dict | None] = {}
        self._dispatches: dict[str, dict] = {}
        self.load_aggregate_calls: list[str] = []

    def add_dispatch(
        self, dispatch_id: str, *, commit: str = "abc1234", row_kwargs: dict | None = None, no_aggregate: bool = False
    ):
        self._dispatches[dispatch_id] = {
            "schema_version": "1.3",
            "dispatch_id": dispatch_id,
            "commit_sha": commit,
            "fleet": [],
            "jobs": [],
            "skipped": [],
        }
        if no_aggregate:
            self._aggregates[dispatch_id] = None
        elif row_kwargs is None:
            self._aggregates[dispatch_id] = {"schema_version": "1.0", "rows": []}
        else:
            self._aggregates[dispatch_id] = {
                "schema_version": "1.0",
                "rows": [_row(**row_kwargs)],
            }

    def load_aggregate(self, dispatch_id: str):
        self.load_aggregate_calls.append(dispatch_id)
        return self._aggregates.get(dispatch_id)

    def load_dispatch(self, dispatch_id: str):
        return self._dispatches[dispatch_id]


def test_compute_points_returns_one_per_dispatch():
    data = _StubData()
    for did in ["d1", "d2", "d3"]:
        data.add_dispatch(did, commit=did, row_kwargs={"agg": _aggregate_block()})
    points = compute_trend_points(data, ["d1", "d2", "d3"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert len(points) == 3


def test_compute_points_skips_missing_aggregate():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={"agg": _aggregate_block()})
    data.add_dispatch("d2", no_aggregate=True)
    data.add_dispatch("d3", row_kwargs={"agg": _aggregate_block()})
    points = compute_trend_points(data, ["d1", "d2", "d3"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert len(points) == 2
    assert [p["dispatch_id"] for p in points] == ["d1", "d3"]


def test_compute_points_skips_dispatch_missing_row():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={"agg": _aggregate_block()})
    data.add_dispatch("d2")  # row_kwargs=None → empty rows
    points = compute_trend_points(data, ["d1", "d2"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert len(points) == 1
    assert points[0]["dispatch_id"] == "d1"


def test_compute_points_uses_aggregate_for_known_metric():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={"agg": _aggregate_block(reward_mean=7000, reward_std=300)})
    points = compute_trend_points(data, ["d1"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert points[0]["mean"] == 7000.0
    assert points[0]["std"] == 300.0


def test_compute_points_computes_from_seeds_for_startup_metric():
    data = _StubData()
    data.add_dispatch(
        "d1",
        row_kwargs={
            "agg": _aggregate_block(),
            "seeds": {
                "42": _seed(startup_app_launch_s=3.0),
                "43": _seed(startup_app_launch_s=4.0),
                "44": _seed(startup_app_launch_s=5.0),
            },
        },
    )
    points = compute_trend_points(data, ["d1"], "X", "rsl_rl", "physx", "startup_app_launch_s")
    assert points[0]["mean"] == 4.0
    # population std of [3, 4, 5] = sqrt(2/3) ≈ 0.8165
    assert abs(points[0]["std"] - 0.8164965809277260) < 1e-6


def test_compute_points_carries_commit_sha():
    data = _StubData()
    data.add_dispatch("d1", commit="abc1234567890", row_kwargs={"agg": _aggregate_block()})
    points = compute_trend_points(data, ["d1"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert points[0]["commit_sha"] == "abc1234567890"


def test_render_trend_chart_ribbon_mode():
    points = [
        {"dispatch_id": "d1", "commit_sha": "aaa1111", "mean": 7000.0, "std": 300.0, "n_seeds_completed": 3},
        {"dispatch_id": "d2", "commit_sha": "bbb2222", "mean": 7500.0, "std": 280.0, "n_seeds_completed": 3},
        {"dispatch_id": "d3", "commit_sha": "ccc3333", "mean": 7991.5, "std": 257.7, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Reward (final EMA)", mode="ribbon")
    fig = graph.figure
    # ribbon mode → 2 traces (mean line + fill band).
    assert len(fig.data) == 2


def test_render_trend_chart_bars_mode():
    points = [
        {"dispatch_id": "d1", "commit_sha": "aaa1111", "mean": 7000.0, "std": 300.0, "n_seeds_completed": 3},
        {"dispatch_id": "d2", "commit_sha": "bbb2222", "mean": 7500.0, "std": 280.0, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Reward (final EMA)", mode="bars")
    fig = graph.figure
    # bars mode → 1 trace with error_y populated.
    assert len(fig.data) == 1
    assert fig.data[0].error_y is not None


def test_render_trend_chart_x_labels_short_sha():
    points = [
        {"dispatch_id": "d1", "commit_sha": "abcdef1234567", "mean": 1.0, "std": 0.1, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Iter time", mode="ribbon")
    fig = graph.figure
    # Tick labels should use the short SHA (first 7).
    tick_text = list(fig.layout.xaxis.ticktext or [])
    assert any("abcdef1" in text for text in tick_text)


def test_render_trend_chart_current_marker_highlighted():
    points = [
        {"dispatch_id": "d1", "commit_sha": "aaa1111", "mean": 7000.0, "std": 300.0, "n_seeds_completed": 3},
        {"dispatch_id": "d2", "commit_sha": "bbb2222", "mean": 7500.0, "std": 280.0, "n_seeds_completed": 3},
        {"dispatch_id": "d3", "commit_sha": "ccc3333", "mean": 7991.5, "std": 257.7, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Reward (final EMA)", mode="ribbon")
    fig = graph.figure
    # Locate the line+marker trace (the second one in ribbon mode).
    line_trace = fig.data[1]
    marker_colors = list(line_trace.marker.color)
    # Last marker color is NVIDIA green (#76b900); others are #66b6ff.
    assert marker_colors[-1] == "#76b900"
    assert all(c == "#66b6ff" for c in marker_colors[:-1])


def test_render_trend_section_empty_state():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    class _Data:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return []

        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("Isaac-Ant-Direct-v0", "rsl_rl", "physx")
    component = render_trend_section(
        _Data(),
        current_dispatch_id="d-now",
        selection=selection,
        metric="reward_final_ema",
        mode="ribbon",
    )
    text_parts = [getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(text_parts)
    assert "Trend needs at least 1 prior" in text


def test_render_trend_section_single_point_state():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    data = _StubData()
    data.add_dispatch("d-now", row_kwargs={"agg": _aggregate_block()})

    class _DataWrapper:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return ["d-now"]

        def load_aggregate(self, dispatch_id):
            return data.load_aggregate(dispatch_id)

        def load_dispatch(self, dispatch_id):
            return data.load_dispatch(dispatch_id)

        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("X", "rsl_rl", "physx")
    component = render_trend_section(
        _DataWrapper(),
        current_dispatch_id="d-now",
        selection=selection,
        metric="reward_final_ema",
        mode="ribbon",
    )
    text_parts = [getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(text_parts)
    assert "First matching dispatch" in text


def test_render_trend_section_normal_render():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    data = _StubData()
    for did in ["d1", "d2", "d3"]:
        data.add_dispatch(did, commit=did, row_kwargs={"agg": _aggregate_block()})

    class _DataWrapper:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return ["d1", "d2", "d3"]

        def load_aggregate(self, dispatch_id):
            return data.load_aggregate(dispatch_id)

        def load_dispatch(self, dispatch_id):
            return data.load_dispatch(dispatch_id)

        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("X", "rsl_rl", "physx")
    component = render_trend_section(
        _DataWrapper(),
        current_dispatch_id="d3",
        selection=selection,
        metric="reward_final_ema",
        mode="ribbon",
    )
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    assert len(graphs) == 1


def test_render_trend_section_places_current_on_the_right():
    """data.trend_dispatches_for returns newest-first; the chart must read
    left=oldest → right=newest so the current dispatch lands on the rightmost
    tick (with the ▲ marker and NVIDIA-green color)."""
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    data = _StubData()
    for did, sha in [("d-now", "ccc3333"), ("d-mid", "bbb2222"), ("d-old", "aaa1111")]:
        data.add_dispatch(did, commit=sha, row_kwargs={"agg": _aggregate_block()})

    class _DataWrapper:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            # Production order: newest-first.
            return ["d-now", "d-mid", "d-old"]

        def load_aggregate(self, dispatch_id):
            return data.load_aggregate(dispatch_id)

        def load_dispatch(self, dispatch_id):
            return data.load_dispatch(dispatch_id)

        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("X", "rsl_rl", "physx")
    component = render_trend_section(
        _DataWrapper(),
        current_dispatch_id="d-now",
        selection=selection,
        metric="reward_final_ema",
        mode="ribbon",
    )
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    # Tick labels read left-to-right; the rightmost gets the ' ▲' suffix and
    # corresponds to the current dispatch (newest, "d-now" → "ccc3333").
    tick_text = list(fig.layout.xaxis.ticktext or [])
    assert tick_text[0].startswith("aaa1111")
    assert tick_text[-1].startswith("ccc3333")
    assert tick_text[-1].endswith(" ▲")
