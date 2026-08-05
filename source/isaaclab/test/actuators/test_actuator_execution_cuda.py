# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA-graph contracts for articulation-owned actuator execution plans."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import warp as wp
from test_actuator_execution import _dc_cfg, _ideal_cfg, _implicit_cfg, _make_plan

from isaaclab.actuators.actuator_net import ActuatorNetMLP
from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
    pytest.mark.filterwarnings("ignore:.*torch.jit.*:DeprecationWarning"),
]


@dataclass(frozen=True)
class _CudaPlan:
    """One real CUDA execution plan and its articulation facade."""

    execution: object
    articulation: object
    control: object
    stream: torch.cuda.Stream

    def warmup_and_capture(self) -> None:
        with torch.cuda.stream(self.stream):
            self.execution.warmup_and_capture()

    def compute(self) -> None:
        with torch.cuda.stream(self.stream):
            self.articulation.compute()


def _mlp_cfg(tmp_path) -> ActuatorNetMLPCfg:
    """Create a tiny real TorchScript actuator network for an eager barrier."""
    network_file = tmp_path / "actuator_net.pt"
    torch.jit.trace(torch.nn.Linear(2, 1, device="cuda"), torch.zeros((1, 2), device="cuda")).save(str(network_file))
    return ActuatorNetMLPCfg(
        joint_names_expr=["ankle"],
        effort_limit=15.0,
        velocity_limit=10.0,
        saturation_effort=25.0,
        network_file=str(network_file),
        pos_scale=1.0,
        vel_scale=1.0,
        torque_scale=1.0,
        input_order="pos_vel",
        input_idx=(0,),
    )


def _make_cuda_plan(*, actuator_types: tuple[type, ...], tmp_path) -> _CudaPlan:
    """Build a CUDA plan whose groups occupy distinct articulation joints."""
    factories = {
        ImplicitActuator: lambda: _implicit_cfg(["hip"]),
        IdealPDActuator: lambda: _ideal_cfg(["knee"]),
        DCMotor: lambda: _dc_cfg(["ankle"]),
        ActuatorNetMLP: lambda: _mlp_cfg(tmp_path),
    }
    groups = {f"group_{index}": factories[actuator_type]() for index, actuator_type in enumerate(actuator_types)}
    _, articulation, control = _make_plan(groups, device="cuda")
    return _CudaPlan(articulation._execution_plan, articulation, control, torch.cuda.Stream(device="cuda"))


def _fill_canonical_inputs(plan: _CudaPlan, *, command: float, position: float, velocity: float) -> None:
    """Write distinct real Torch values into every stable CUDA input allocation."""
    with torch.cuda.stream(plan.stream):
        plan.articulation.command.position.torch.fill_(command)
        plan.articulation.command.velocity.torch.fill_(command)
        plan.articulation.command.effort.torch.fill_(command)
        plan.control.joint_pos.torch.fill_(position)
        plan.control.joint_vel.torch.fill_(velocity)


def _all_plan_pointers(plan: _CudaPlan) -> tuple[int, ...]:
    """Return storage pointers that graph replay must retain."""
    pointers = [
        plan.articulation.command.position.torch.data_ptr(),
        plan.articulation.command.velocity.torch.data_ptr(),
        plan.articulation.command.effort.torch.data_ptr(),
        plan.articulation.joint_command.position.torch.data_ptr(),
        plan.articulation.joint_command.velocity.torch.data_ptr(),
        plan.articulation.joint_command.effort.torch.data_ptr(),
        plan.articulation.computed_effort.torch.data_ptr(),
        plan.articulation.applied_effort.torch.data_ptr(),
        plan.control.joint_pos.torch.data_ptr(),
        plan.control.joint_vel.torch.data_ptr(),
    ]
    for execution_range in plan.execution.stateless_ranges:
        pointers.extend(value.torch.data_ptr() for value in execution_range.staging.values())
        pointers.extend(
            value.torch.data_ptr() for value in execution_range.executor.__dict__["_parameter_binding"].arrays.values()
        )
    return tuple(pointers)


def _clone_applied_effort(plan: _CudaPlan) -> torch.Tensor:
    """Clone graph output on the same non-default Torch/Warp stream."""
    with torch.cuda.stream(plan.stream):
        result = plan.articulation.applied_effort.torch.clone()
    torch.cuda.current_stream().wait_stream(plan.stream)
    return result


def _assert_matches_eager_literal(value: torch.Tensor, *, command: float, position: float, velocity: float) -> None:
    """Check the hand-derived literal outputs for the three supported stateless types."""
    expected_implicit = 2.0 * (command - position) + 0.5 * (command - velocity) + command
    expected_ideal = 2.0 * (command - position) + 0.5 * (command - velocity) + command
    computed_dc = 2.0 * (command - position) + 0.5 * (command - velocity) + command
    expected_dc = min(max(computed_dc, -15.0), 15.0)
    expected = torch.tensor([[expected_implicit, expected_ideal, expected_dc]] * 2, device="cuda")
    torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def test_warp_torch_interop_capture_uses_changed_pointer_stable_inputs(tmp_path) -> None:
    """Catch a graph that replays stale Torch inputs or replaces graph-owned storage."""
    plan = _make_cuda_plan(actuator_types=(ImplicitActuator, IdealPDActuator, DCMotor), tmp_path=tmp_path)
    plan.warmup_and_capture()
    first_pointers = _all_plan_pointers(plan)
    _fill_canonical_inputs(plan, command=1.0, position=0.25, velocity=-0.5)
    plan.compute()
    first = _clone_applied_effort(plan)
    _fill_canonical_inputs(plan, command=3.0, position=-0.75, velocity=0.5)
    plan.compute()
    second = _clone_applied_effort(plan)
    assert not torch.equal(first, second)
    assert _all_plan_pointers(plan) == first_pointers
    _assert_matches_eager_literal(second, command=3.0, position=-0.75, velocity=0.5)


def test_default_torch_stream_uses_a_dedicated_graph_stream(monkeypatch, tmp_path) -> None:
    """Catch default-stream callers that silently lose graph capture or ordering."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    warmed_streams = []
    original_warmup = plan.execution._warmup_graphable_prefix

    def _record_warmup() -> None:
        warmed_streams.append(torch.cuda.current_stream().cuda_stream)
        original_warmup()

    monkeypatch.setattr(plan.execution, "_warmup_graphable_prefix", _record_warmup)
    plan.execution.warmup_and_capture()
    assert plan.execution._full_graph is not None
    assert plan.execution._graph_torch_stream is not None
    assert plan.execution._graph_torch_stream.cuda_stream != torch.cuda.current_stream().cuda_stream
    assert warmed_streams == [plan.execution._graph_torch_stream.cuda_stream]
    original_event = torch.cuda.Event

    def _forbid_event(*args, **kwargs):
        raise AssertionError("replay allocated a CUDA event")

    def _forbid_wait_stream(*args, **kwargs):
        raise AssertionError("replay used an allocating wait_stream barrier")

    monkeypatch.setattr(torch.cuda, "Event", _forbid_event)
    monkeypatch.setattr(torch.cuda.Stream, "wait_stream", _forbid_wait_stream)
    plan.articulation.command.position.torch.fill_(3.0)
    plan.articulation.command.velocity.torch.fill_(3.0)
    plan.articulation.command.effort.torch.fill_(3.0)
    plan.control.joint_pos.torch.fill_(-0.75)
    plan.control.joint_vel.torch.fill_(0.5)
    plan.articulation.compute()
    expected = min(max(2.0 * 3.75 + 0.5 * 2.5 + 3.0, -15.0), 15.0)
    torch.testing.assert_close(plan.articulation.applied_effort.torch[:, 2], torch.full((2,), expected, device="cuda"))
    monkeypatch.setattr(torch.cuda, "Event", original_event)


def test_fully_graphable_plan_replays_one_complete_graph(monkeypatch, tmp_path) -> None:
    """Catch a graphable plan that still dispatches its eager scatter sequence."""
    plan = _make_cuda_plan(actuator_types=(ImplicitActuator, IdealPDActuator, DCMotor), tmp_path=tmp_path)
    plan.warmup_and_capture()
    assert plan.execution._full_graph is not None
    assert plan.execution._prefix_graph is None
    replayed = []
    original_capture_launch = wp.capture_launch
    original_scatter = plan.execution._scatter_static_epoch

    def _record_capture_launch(graph, *args, **kwargs) -> None:
        replayed.append(graph)
        original_capture_launch(graph, *args, **kwargs)

    def _reject_eager_scatter(*args, **kwargs) -> None:
        raise AssertionError("full graph replay dispatched eager scatter")

    monkeypatch.setattr(wp, "capture_launch", _record_capture_launch)
    monkeypatch.setattr(plan.execution, "_scatter_static_epoch", _reject_eager_scatter)
    plan.compute()
    assert replayed == [plan.execution._full_graph]
    monkeypatch.setattr(plan.execution, "_scatter_static_epoch", original_scatter)


def test_mixed_plan_captures_prefix_then_runs_eager_and_cached_scatter(monkeypatch, tmp_path) -> None:
    """Catch mixed execution that omits its eager barrier or fails to replay the graphable prefix."""
    plan = _make_cuda_plan(actuator_types=(IdealPDActuator, ActuatorNetMLP), tmp_path=tmp_path)
    plan.warmup_and_capture()
    assert plan.execution._full_graph is None
    assert plan.execution._prefix_graph is not None
    replayed = []
    eager_groups = []
    cached_scatter = []
    original_capture_launch = wp.capture_launch
    original_eager = plan.execution._run_eager
    original_scatter = plan.execution._scatter_static_epoch

    def _record_capture_launch(graph, *args, **kwargs) -> None:
        replayed.append(graph)
        original_capture_launch(graph, *args, **kwargs)

    def _record_eager(segment) -> None:
        eager_groups.append(segment.group_name)
        original_eager(segment)

    def _record_scatter(epoch) -> None:
        cached_scatter.append(epoch.group_names)
        original_scatter(epoch)

    monkeypatch.setattr(wp, "capture_launch", _record_capture_launch)
    monkeypatch.setattr(plan.execution, "_run_eager", _record_eager)
    monkeypatch.setattr(plan.execution, "_scatter_static_epoch", _record_scatter)
    plan.compute()
    assert replayed == [plan.execution._prefix_graph]
    assert len(eager_groups) == 1
    assert len(cached_scatter) == 1


def test_mixed_prefix_refreshes_registered_projection(tmp_path) -> None:
    """Catch a prefix replay that returns before its compatibility epilogue."""
    plan = _make_cuda_plan(actuator_types=(IdealPDActuator, ActuatorNetMLP), tmp_path=tmp_path)
    plan.warmup_and_capture()
    assert plan.execution._prefix_graph is not None
    held = plan.articulation._get_compatibility_projection("soft_joint_vel_limits").torch
    with torch.cuda.stream(plan.stream):
        plan.articulation["group_0"].velocity_limit.fill_(7.0)
    plan.compute()
    torch.cuda.current_stream().wait_stream(plan.stream)
    expected = torch.zeros_like(held)
    expected[:, 1].fill_(7.0)
    expected[:, 2].fill_(10.0)
    torch.testing.assert_close(held, expected, rtol=0.0, atol=0.0)


def test_capture_failure_falls_back_for_the_generation(monkeypatch, tmp_path) -> None:
    """Catch a failed capture retried on every frame instead of using cached eager launches."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    attempts = 0
    eager_steps = 0
    original_eager_step = plan.execution._run_eager_sequence

    def _fail_capture_begin(*args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("test capture failure")

    def _record_eager_step(dt: float = 0.0) -> None:
        nonlocal eager_steps
        eager_steps += 1
        original_eager_step(dt)

    monkeypatch.setattr(wp, "capture_begin", _fail_capture_begin)
    monkeypatch.setattr(plan.execution, "_run_eager_sequence", _record_eager_step)
    with pytest.warns(RuntimeWarning, match="cached eager execution"):
        plan.warmup_and_capture()
    plan.compute()
    plan.compute()
    assert plan.execution._graph_capture_failed
    assert attempts == 1
    assert eager_steps == 2


def test_capture_failure_eager_fallback_refreshes_registered_projection(monkeypatch, tmp_path) -> None:
    """Catch graph teardown that drops a registered projection's eager refresh route."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    held = plan.articulation._get_compatibility_projection("soft_joint_vel_limits").torch

    def _fail_capture_begin(*args, **kwargs) -> None:
        raise RuntimeError("test capture failure")

    monkeypatch.setattr(wp, "capture_begin", _fail_capture_begin)
    with pytest.warns(RuntimeWarning, match="cached eager execution"):
        plan.warmup_and_capture()
    with torch.cuda.stream(plan.stream):
        plan.articulation["group_0"].velocity_limit.fill_(7.0)
    plan.compute()
    torch.cuda.current_stream().wait_stream(plan.stream)
    expected = torch.zeros_like(held)
    expected[:, 2].fill_(7.0)
    torch.testing.assert_close(held, expected, rtol=0.0, atol=0.0)


def test_projection_activated_after_capture_refreshes_outside_graph(tmp_path) -> None:
    """Catch a late compatibility view that invalidates or leaves a full graph stale."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    plan.warmup_and_capture()
    assert plan.execution._post_graph_projection_launches == ()
    held = plan.articulation._get_compatibility_projection("soft_joint_vel_limits").torch
    assert any(key[0] == "compatibility_fill" for key in plan.execution._launch_cache._commands)
    with torch.cuda.stream(plan.stream):
        plan.articulation["group_0"].velocity_limit.fill_(7.0)
    plan.compute()
    torch.cuda.current_stream().wait_stream(plan.stream)
    expected = torch.zeros_like(held)
    expected[:, 2].fill_(7.0)
    torch.testing.assert_close(held, expected, rtol=0.0, atol=0.0)
    assert len(plan.execution._post_graph_projection_launches) == 1
    assert plan.execution._full_graph is not None


def test_outer_capture_eager_fallback_refreshes_registered_projection(tmp_path) -> None:
    """Catch outer capture fallback that leaves a full-graph projection stale."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    plan.warmup_and_capture()
    held = plan.articulation._get_compatibility_projection("soft_joint_vel_limits").torch
    with torch.cuda.stream(plan.stream):
        plan.articulation["group_0"].velocity_limit.fill_(7.0)
        stream = wp.stream_from_torch(plan.stream)
        with wp.ScopedStream(stream, sync_enter=False):
            with wp.ScopedCapture(stream=stream) as capture:
                plan.compute()
        wp.capture_launch(capture.graph, stream=stream)
    torch.cuda.current_stream().wait_stream(plan.stream)
    expected = torch.zeros_like(held)
    expected[:, 2].fill_(7.0)
    torch.testing.assert_close(held, expected, rtol=0.0, atol=0.0)


def test_plan_invalidation_releases_graphs_and_recorded_launches(tmp_path) -> None:
    """Catch generation teardown retaining a replayable graph or eager launch command."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    plan.warmup_and_capture()
    assert plan.execution._full_graph is not None
    assert plan.execution._launch_cache._commands
    plan.execution.invalidate()
    assert plan.execution._full_graph is None
    assert plan.execution._prefix_graph is None
    assert plan.execution._post_graph_projection_launches == ()
    assert plan.execution._launch_cache._commands == {}


def test_invalidation_retries_failed_graph_lease_cleanup(monkeypatch, tmp_path) -> None:
    """Catch a failed graph revocation that permanently pins graph-owned storage."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    plan.warmup_and_capture()
    lease = plan.execution._full_graph_lease
    assert lease is not None
    original_revoke = lease.revoke
    attempts = 0

    def _fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("test graph cleanup failure")
        original_revoke()

    monkeypatch.setattr(lease, "revoke", _fail_once)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        plan.execution.invalidate()
    assert plan.execution._valid
    assert plan.execution._retired_graph_leases == [lease]
    plan.execution.invalidate()
    assert attempts == 2
    assert not plan.execution._valid


def test_graph_replay_matches_eager_all_stateless_types_exactly(tmp_path) -> None:
    """Catch a captured stateless sequence that changes established eager arithmetic."""
    graph_plan = _make_cuda_plan(actuator_types=(ImplicitActuator, IdealPDActuator, DCMotor), tmp_path=tmp_path)
    eager_plan = _make_cuda_plan(actuator_types=(ImplicitActuator, IdealPDActuator, DCMotor), tmp_path=tmp_path)
    _fill_canonical_inputs(graph_plan, command=3.0, position=-0.75, velocity=0.5)
    _fill_canonical_inputs(eager_plan, command=3.0, position=-0.75, velocity=0.5)
    graph_plan.warmup_and_capture()
    graph_plan.compute()
    eager_plan.articulation.compute()
    torch.cuda.current_stream().wait_stream(graph_plan.stream)
    torch.testing.assert_close(
        graph_plan.articulation.applied_effort.torch,
        eager_plan.articulation.applied_effort.torch,
        rtol=0.0,
        atol=0.0,
    )


def test_capture_end_failure_marks_generation_eager_once(monkeypatch, tmp_path) -> None:
    """Catch a post-begin failure that leaves a graph live or retries capture later."""
    plan = _make_cuda_plan(actuator_types=(DCMotor,), tmp_path=tmp_path)
    original_capture_end = wp.capture_end
    end_calls = 0

    def _fail_after_end(*args, **kwargs):
        nonlocal end_calls
        end_calls += 1
        graph = original_capture_end(*args, **kwargs)
        if end_calls == 1:
            raise RuntimeError("test capture-end failure")
        return graph

    monkeypatch.setattr(wp, "capture_end", _fail_after_end)
    with pytest.warns(RuntimeWarning, match="cached eager execution"):
        plan.warmup_and_capture()
    plan.warmup_and_capture()
    assert plan.execution._graph_capture_failed
    assert plan.execution._full_graph is None
    assert plan.execution._prefix_graph is None
    assert plan.execution._post_graph_projection_launches == ()
    assert end_calls == 2
