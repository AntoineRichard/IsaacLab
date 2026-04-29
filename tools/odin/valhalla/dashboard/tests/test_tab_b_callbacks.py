# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab B callback helpers."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
    _compute_picker_children,
    _serialize_to_url,
    _sync_url_to_selection,
)


class _StubData:
    """Drop-in DataLayer for callback tests."""

    def __init__(self, aggregate=None):
        self._agg = aggregate

    def load_aggregate(self, dispatch_id: str):
        return self._agg


def _aggregate(rows: list[dict]) -> dict:
    return {"schema_version": "1.0", "rows": rows}


def _row(task: str, framework: str = "rsl_rl", backend: str = "physx") -> dict:
    return {
        "task": task,
        "framework": framework,
        "backend": backend,
        "aggregate": {},
        "seeds": {},
        "divergent_seeds": [],
    }


def test_init_picker_returns_picker_div():
    data = _StubData(aggregate=_aggregate([_row("X")]))
    out = _compute_picker_children(data, "d-1", search="?task=X&framework=rsl_rl&backend=physx")
    assert getattr(out, "id", None) == "tab-b-picker-content"


def test_sync_url_to_selection_parses_full():
    out = _sync_url_to_selection("?task=A&framework=rsl_rl&backend=physx")
    assert out == "A|rsl_rl|physx"


def test_sync_url_to_selection_handles_empty():
    assert _sync_url_to_selection("") is None


def test_picker_to_url_serializes_value():
    out = _serialize_to_url("A|rsl_rl|physx")
    assert out == "?task=A&framework=rsl_rl&backend=physx"


def test_update_curves_and_stats_loads_three_seed_bundles():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_curves_and_stats,
    )

    aggregate = _aggregate(
        [
            {
                "task": "X",
                "framework": "rsl_rl",
                "backend": "physx",
                "aggregate": {"reward_final_ema": {"mean": 1, "std": 0, "min": 0, "max": 0, "cv_pct": 0}},
                "seeds": {
                    "42": {"run_id": "rsl-rl_physx_X_seed42", "status": "completed"},
                    "43": {"run_id": "rsl-rl_physx_X_seed43", "status": "completed"},
                    "44": {"run_id": "rsl-rl_physx_X_seed44", "status": "completed"},
                },
                "divergent_seeds": [],
            },
        ]
    )

    class _Data:
        def __init__(self):
            self.load_training_calls: list = []

        def load_aggregate(self, dispatch_id):
            return aggregate

        def load_training(self, dispatch_id, run_id):
            self.load_training_calls.append((dispatch_id, run_id))
            return

    data = _Data()
    curves, stats = _compute_curves_and_stats(
        data,
        dispatch_id="d-1",
        selection_value="X|rsl_rl|physx",
    )
    assert len(data.load_training_calls) == 3
    assert getattr(curves, "id", None) is not None
    assert getattr(stats, "id", None) is not None


def test_update_curves_and_stats_returns_curves_and_stats_divs():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_curves_and_stats,
    )

    aggregate = _aggregate([_row("X")])

    class _Data:
        def load_aggregate(self, dispatch_id):
            return aggregate

        def load_training(self, dispatch_id, run_id):
            return None

    curves, stats = _compute_curves_and_stats(
        _Data(),
        dispatch_id="d-1",
        selection_value="X|rsl_rl|physx",
    )
    # The IDs are 'tab-b-curves-content' and 'tab-b-stats-content'.
    assert getattr(curves, "id", None) == "tab-b-curves-content"
    assert getattr(stats, "id", None) == "tab-b-stats-content"


def test_update_curves_and_stats_renders_banner_when_row_missing():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_curves_and_stats,
    )

    aggregate = _aggregate([_row("Y")])  # row "X" not in aggregate

    class _Data:
        def load_aggregate(self, dispatch_id):
            return aggregate

        def load_training(self, dispatch_id, run_id):
            return None

    curves, stats = _compute_curves_and_stats(
        _Data(),
        dispatch_id="d-1",
        selection_value="X|rsl_rl|physx",
    )

    def _walk(c):
        yield c
        kids = getattr(c, "children", None)
        if isinstance(kids, list):
            for k in kids:
                if k is not None and not isinstance(k, str):
                    yield from _walk(k)
        elif kids is not None and not isinstance(kids, str):
            yield from _walk(kids)

    text = (
        " ".join(
            str(getattr(c, "children", "") or "")
            for c in _walk(curves)
            if isinstance(getattr(c, "children", None), str)
        )
        + " "
        + " ".join(
            str(getattr(c, "children", "") or "") for c in _walk(stats) if isinstance(getattr(c, "children", None), str)
        )
    )
    assert "Row not found" in text


def test_update_trend_returns_trend_div():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_trend_children,
    )

    class _Data:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return []

        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    out = _compute_trend_children(
        _Data(),
        dispatch_id="d-1",
        selection_value="X|rsl_rl|physx",
        metric="reward_final_ema",
        mode="ribbon",
    )
    assert getattr(out, "id", None) == "tab-b-trend-content"


def test_update_trend_ignores_phantom_initial_calls():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_trend_children,
    )

    class _Data:
        pass

    # No selection_value → returns the empty-state div with no Graph.
    out = _compute_trend_children(
        _Data(),
        dispatch_id="d-1",
        selection_value=None,
        metric="reward_final_ema",
        mode="ribbon",
    )
    graphs = []

    def _walk(c):
        yield c
        kids = getattr(c, "children", None)
        if isinstance(kids, list):
            for k in kids:
                if k is not None and not isinstance(k, str):
                    yield from _walk(k)
        elif kids is not None and not isinstance(kids, str):
            yield from _walk(kids)

    for c in _walk(out):
        if type(c).__name__ == "Graph":
            graphs.append(c)
    assert len(graphs) == 0
