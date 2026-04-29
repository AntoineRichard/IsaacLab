# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.curves."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.curves import (
    _SEED_PALETTE,
    render_curves,
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


def _bundle(reward: list[float] | None = None, ep_length: list[float] | None = None) -> dict:
    payload: dict = {"schema_version": "1.0", "learning": {}}
    if reward is not None:
        payload["learning"]["reward"] = {"series_per_iter": reward, "final_ema": reward[-1]}
    if ep_length is not None:
        payload["learning"]["ep_length"] = {"series_per_iter": ep_length, "final_ema": ep_length[-1]}
    return payload


def _series(start: float, n: int = 10, slope: float = 5.0) -> list[float]:
    return [start + i * slope for i in range(n)]


def test_render_curves_returns_two_graph_components():
    bundles = {
        "42": _bundle(reward=_series(10), ep_length=_series(100)),
        "43": _bundle(reward=_series(15), ep_length=_series(110)),
    }
    component = render_curves(bundles, divergent_seeds=[])
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    assert len(graphs) == 2


def test_render_curves_one_trace_per_seed():
    bundles = {f"4{i}": _bundle(reward=_series(10 + i)) for i in range(3)}
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    assert len(fig.data) == 3


def test_render_curves_seed_color_assignment_deterministic():
    bundles = {
        "42": _bundle(reward=_series(10)),
        "43": _bundle(reward=_series(15)),
        "44": _bundle(reward=_series(20)),
    }
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    colors = [trace.line.color for trace in fig.data]
    assert colors[:3] == _SEED_PALETTE[:3]


def test_render_curves_divergent_seed_styled_differently():
    bundles = {
        "42": _bundle(reward=_series(10)),
        "43": _bundle(reward=_series(15)),
    }
    component = render_curves(bundles, divergent_seeds=["43"])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    seed_43_trace = next(t for t in fig.data if t.name == "seed 43")
    assert seed_43_trace.line.color == "#e16868"
    assert seed_43_trace.line.dash == "dash"


def test_render_curves_handles_missing_seed_series():
    bundles = {
        "42": _bundle(reward=_series(10)),
        "43": _bundle(),  # no series
    }
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    # Only seed 42 should have a trace.
    assert len(fig.data) == 1
    assert fig.data[0].name == "seed 42"


def test_render_curves_empty_bundles_renders_empty_state():
    component = render_curves({}, divergent_seeds=[])
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    assert len(graphs) == 0
    text_parts = [getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(text_parts)
    assert "No training.json bundles" in text


def test_render_curves_x_axis_label():
    bundles = {"42": _bundle(reward=_series(10))}
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    assert "iteration" in fig.layout.xaxis.title.text.lower()


def test_render_curves_reward_y_axis_label():
    bundles = {"42": _bundle(reward=_series(10))}
    component = render_curves(bundles, divergent_seeds=[])
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    reward_graph = graphs[0]
    fig = reward_graph.figure
    assert "reward" in fig.layout.yaxis.title.text.lower()
