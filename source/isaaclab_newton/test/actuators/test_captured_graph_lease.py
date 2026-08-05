# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for private revocable actuator CUDA-graph ownership."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def _fake_graph() -> SimpleNamespace:
    return SimpleNamespace(
        device=SimpleNamespace(context=object(), context_guard=nullcontext()),
        graph=object(),
        graph_exec=object(),
    )


def test_captured_graph_lease_revokes_retained_reference_exactly_once(monkeypatch) -> None:
    """A retained lease rejects replay after deterministic native destruction."""
    import isaaclab_newton.actuators._graph as graph_module

    events = []
    graph = _fake_graph()
    raw_graph, raw_exec = graph.graph, graph.graph_exec
    monkeypatch.setattr(
        graph_module.wp, "capture_launch", lambda value, stream=None: events.append(("launch", value, stream))
    )
    monkeypatch.setattr(graph_module.wp, "synchronize_device", lambda device: events.append(("sync", device)))
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda context, value: events.append(("destroy_graph", context, value)) or True,
                wp_cuda_graph_exec_destroy=lambda context, value: events.append(("destroy_exec", context, value))
                or True,
            )
        ),
    )
    lease = graph_module._CapturedGraphLease(graph, generation=object(), label="native actuator graph")

    lease.launch(stream="stream")
    lease.revoke()
    lease.revoke()

    assert lease.is_live is False
    assert graph.graph is None
    assert graph.graph_exec is None
    assert [event[0] for event in events] == ["launch", "sync", "destroy_graph", "destroy_exec"]
    assert events[2][2] is raw_graph
    assert events[3][2] is raw_exec
    with pytest.raises(RuntimeError, match="native actuator graph.*revoked"):
        lease.launch()


def test_captured_graph_lease_preserves_primary_cleanup_error(monkeypatch) -> None:
    """Revocation stays effective and annotates later native cleanup failures."""
    import isaaclab_newton.actuators._graph as graph_module

    graph = _fake_graph()
    primary = RuntimeError("graph destroy failed")
    secondary = RuntimeError("exec destroy failed")
    monkeypatch.setattr(graph_module.wp, "synchronize_device", lambda _device: None)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda *_args: (_ for _ in ()).throw(primary),
                wp_cuda_graph_exec_destroy=lambda *_args: (_ for _ in ()).throw(secondary),
            )
        ),
    )
    lease = graph_module._CapturedGraphLease(graph, generation=object(), label="hosted actuator graph")

    with pytest.raises(RuntimeError, match="graph destroy failed") as raised:
        lease.revoke()

    assert lease.is_live is False
    assert any("exec destroy failed" in note for note in getattr(raised.value, "__notes__", ()))
    with pytest.raises(RuntimeError, match="hosted actuator graph.*revoked"):
        lease.launch()
    with pytest.raises(RuntimeError, match="graph destroy failed"):
        lease.revoke()


def test_captured_graph_lease_retries_remaining_native_handle_after_partial_failure(monkeypatch) -> None:
    """A failed graph-exec destroy remains retryable while replay is permanently disabled."""
    import isaaclab_newton.actuators._graph as graph_module

    graph = _fake_graph()
    raw_exec = graph.graph_exec
    exec_destroy_attempts = 0

    def destroy_exec(_context, value) -> bool:
        nonlocal exec_destroy_attempts
        assert value is raw_exec
        exec_destroy_attempts += 1
        if exec_destroy_attempts == 1:
            raise RuntimeError("first exec destroy failed")
        return True

    monkeypatch.setattr(graph_module.wp, "synchronize_device", lambda _device: None)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda *_args: True,
                wp_cuda_graph_exec_destroy=destroy_exec,
            )
        ),
    )
    generation = object()
    lease = graph_module._CapturedGraphLease(graph, generation=generation, label="retryable actuator graph")

    with pytest.raises(RuntimeError, match="first exec destroy failed"):
        lease.revoke()

    assert lease.is_live is False
    assert lease._generation is generation
    assert graph.graph is None
    assert graph.graph_exec is raw_exec
    with pytest.raises(RuntimeError, match="retryable actuator graph.*revoked"):
        lease.launch()

    lease.revoke()

    assert exec_destroy_attempts == 2
    assert lease._generation is None
    assert graph.graph_exec is None


def test_captured_graph_lease_retries_handles_when_native_destroy_returns_false(monkeypatch) -> None:
    """A false native destroy result keeps both handles and their captured generation alive."""
    import isaaclab_newton.actuators._graph as graph_module

    graph = _fake_graph()
    raw_graph, raw_exec = graph.graph, graph.graph_exec
    generation = object()
    destroy_attempts = {"graph": 0, "graph_exec": 0}

    def destroy_graph(_context, value) -> bool:
        assert value is raw_graph
        destroy_attempts["graph"] += 1
        return destroy_attempts["graph"] > 1

    def destroy_exec(_context, value) -> bool:
        assert value is raw_exec
        destroy_attempts["graph_exec"] += 1
        return destroy_attempts["graph_exec"] > 1

    monkeypatch.setattr(graph_module.wp, "synchronize_device", lambda _device: None)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=destroy_graph,
                wp_cuda_graph_exec_destroy=destroy_exec,
            )
        ),
    )
    lease = graph_module._CapturedGraphLease(graph, generation=generation, label="false-return actuator graph")

    with pytest.raises(RuntimeError):
        lease.revoke()

    assert lease.is_live is False
    assert lease._generation is generation
    assert graph.graph is raw_graph
    assert graph.graph_exec is raw_exec
    assert destroy_attempts == {"graph": 1, "graph_exec": 1}
    with pytest.raises(RuntimeError, match="false-return actuator graph.*revoked"):
        lease.launch()

    lease.revoke()

    assert lease._generation is None
    assert graph.graph is None
    assert graph.graph_exec is None
    assert destroy_attempts == {"graph": 2, "graph_exec": 2}


def test_captured_graph_lease_defers_destruction_and_retains_generation_when_synchronization_fails(
    monkeypatch,
) -> None:
    """A failed synchronization keeps every captured buffer and native handle alive for retry."""
    import isaaclab_newton.actuators._graph as graph_module

    graph = _fake_graph()
    generation = object()
    events = []
    synchronization_attempts = 0

    def synchronize(_device) -> None:
        nonlocal synchronization_attempts
        synchronization_attempts += 1
        events.append("sync")
        if synchronization_attempts == 1:
            raise RuntimeError("first synchronization failed")

    monkeypatch.setattr(graph_module.wp, "synchronize_device", synchronize)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda *_args: events.append("destroy_graph") or True,
                wp_cuda_graph_exec_destroy=lambda *_args: events.append("destroy_exec") or True,
            )
        ),
    )
    lease = graph_module._CapturedGraphLease(graph, generation=generation, label="synchronized actuator graph")
    raw_graph, raw_exec = graph.graph, graph.graph_exec

    with pytest.raises(RuntimeError, match="first synchronization failed"):
        lease.revoke()

    assert lease.is_live is False
    assert lease._generation is generation
    assert graph.graph is raw_graph
    assert graph.graph_exec is raw_exec
    assert events == ["sync"]

    lease.revoke()

    assert lease._generation is None
    assert graph.graph is None
    assert graph.graph_exec is None
    assert events == ["sync", "sync", "destroy_graph", "destroy_exec"]
