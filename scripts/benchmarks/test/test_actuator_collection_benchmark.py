# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the private actuator-collection benchmark driver."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCHMARK = Path(__file__).parents[1] / "benchmark_actuator_collection.py"
_SUMMARY = Path(__file__).parents[1] / "summarize_actuator_collection.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _record(benchmark, *, status="accepted", timing=None, capability=None):
    return {
        "schema": "actuator_collection_attempt/v1",
        "identity": {
            "batch_id": "batch",
            "observation_key": "B1/key",
            "attempt_id": "attempt-01",
            "candidate_sha": "c" * 40,
            "revision_shas": {"develop": "d" * 40, "global": "g" * 40},
            "harness_sha256": "a" * 64,
        },
        "kind": "pair",
        "status": status,
        "boundary": "resolved_construction_to_first_application",
        "telemetry": {"required": False, "available": True, "samples": [], "rejection_reasons": []},
        "members": [
            {
                "revision": "develop",
                "requested_execution": "cached_eager",
                "effective_execution": "cached_eager",
                "revision_sha": "d" * 40,
                "adapter": "develop",
                "resolved_row": {"case": "B1"},
                "source_emulation": False,
                "capability": capability or {"supported": True, "reason": None},
                "timing": timing or {"samples_ms": [1.0]},
                "counters": {},
                "structural": None,
            },
            {
                "revision": "global",
                "requested_execution": "cached_eager",
                "effective_execution": "cached_eager",
                "revision_sha": "g" * 40,
                "adapter": "global",
                "resolved_row": {"case": "B1"},
                "source_emulation": False,
                "capability": {"supported": True, "reason": None},
                "timing": {"samples_ms": [2.0]},
                "counters": {},
                "structural": {},
            },
        ],
        "paths": {"harness": "/tmp/harness", "worktrees": {}, "cache": {}},
        "command": [],
        "device": "cpu",
        "cache": {"policy": "private"},
        "process": {"returncode": 0},
        "metadata": {},
    }


def test_build_matrix_freezes_b0_through_b8_dimensions():
    """Changing a frozen workload dimension must change the generated rows."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_matrix")
    cases = benchmark.build_matrix()
    assert [case.name for case in cases] == [f"B{i}" for i in range(9)]
    rows = benchmark.expand_build_matrix()
    assert [row.case for row in rows[:3]] == ["B0", "B1", "B1"]
    assert [(row.case, row.num_worlds) for row in rows if row.case == "B1"] == [("B1", 1), ("B1", 64), ("B1", 4096)]
    assert next(row for row in rows if row.case == "B3").num_sources == 4
    assert next(row for row in rows if row.case == "B7").num_worlds == 64


def test_b2_b6_and_b8_are_global_only():
    """Historical adapters must never receive global-only build rows."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_global_only")
    cases = {case.name: case for case in benchmark.build_matrix()}
    assert all(cases[name].global_only for name in ("B2", "B6", "B8"))


def test_runtime_matrix_has_18_requested_rows_and_no_fake_graph_execution():
    """Historical graph requests are capability records rather than eager measurements."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runtime")
    rows = benchmark.runtime_matrix("develop")
    assert len(rows) == 18
    assert [row.requested_execution for row in rows[::2]] == ["cached_eager"] * 9
    graph = [row for row in rows if row.requested_execution == "graph"]
    assert all(row.effective_execution is None for row in graph)


def test_all_selectors_reject_scalar_overrides():
    """All-case scheduling must remain a frozen, unmodified matrix."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_cli_all")
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--mode", "build", "--case", "all", "--num_worlds", "1"])


def test_cli_rejects_nonpositive_and_incomplete_child_identity():
    """A final child cannot write anonymous or zero-work observations."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_cli")
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--mode", "build", "--case", "B1", "--num_iterations", "0"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--mode", "build", "--case", "B1", "--final_run", "--revision", "develop"])


def test_record_rejects_unsupported_timing_and_missing_identity():
    """Unsupported capability evidence cannot be silently converted to samples."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_validation")
    record = _record(benchmark)
    record["members"][0]["capability"] = {"supported": False, "reason": "graph unsupported"}
    record["members"][0]["effective_execution"] = None
    with pytest.raises(ValueError, match="unsupported"):
        benchmark.validate_attempt(record)
    del record["identity"]["attempt_id"]
    with pytest.raises(ValueError, match="attempt_id"):
        benchmark.validate_attempt(record)


def test_select_adapter_feature_detection_is_guarded(monkeypatch):
    """A partial historical collection must report unsupported, not use another API."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_adapter_select")
    monkeypatch.setattr(benchmark, "_import_actuator_collection", lambda: type("Partial", (), {})())
    capability = benchmark.select_adapter("current", device="cpu")
    assert capability.supported is False
    assert "feature" in capability.reason


def test_adapters_receive_identical_workload_values():
    """Adapter selection must preserve driver-owned rows and first commands."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_adapter_values")
    row = benchmark.expand_build_matrix("B1")[0]
    workload = benchmark.make_workload(row, "cpu")
    assert workload.group_values == benchmark.make_workload(row, "cpu").group_values


def test_graph_capture_failure_is_rejected_not_eager():
    """A failed graph capture cannot be relabelled as eager execution."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_capture")
    adapter = benchmark._MemoryAdapter("global", "cpu")
    adapter.warmup_execution = lambda _: False
    result = benchmark.measure_runtime(adapter, benchmark.runtime_matrix("global")[1], 1, 1)
    assert result["status"] == "rejected"
    assert result["effective_execution"] is None
    assert result["counters"]["capture"] is not None
    assert result["counters"]["replay"] is None


def test_cpu_b1_smoke_builds_and_applies_once(tmp_path):
    """The import-safe CPU path produces one valid B1 first application."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_cpu")
    output = tmp_path / "smoke"
    assert (
        benchmark.main(
            [
                "--mode",
                "build",
                "--case",
                "B1",
                "--num_worlds",
                "1",
                "--num_sources",
                "1",
                "--num_articulations",
                "1",
                "--groups",
                "3",
                "--actuator_types",
                "implicit",
                "--warmup_iterations",
                "1",
                "--num_iterations",
                "1",
                "--device",
                "cpu",
                "--benchmark_formatter",
                "schema",
                "--output_path",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads(next(output.glob("*/*/attempt.json")).read_text())
    benchmark.validate_attempt(record)
    assert record["members"][0]["timing"]["first_application_count"] == 1
    assert record["members"][0]["adapter"] == "_GlobalCollectionAdapter"
    assert record["members"][0]["resolved_row"]["groups"] == 3


def test_global_introspection_deduplicates_literal_owners():
    """Two aliases to one allocation are reported as one canonical owner."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_introspection")
    owner = type("Owner", (), {"device": "cuda:0", "ptr": 3, "nbytes": 24})()
    data = benchmark._GlobalIntrospector().inspect(type("Generation", (), {"stores": [owner, owner], "plans": []})())
    assert data["canonical_allocation_count"] == 1
    assert data["canonical_allocation_bytes"] == 24
    assert data["storage_wrapper_count"] == 1


def test_global_introspection_counts_alias_descriptors_and_fixed_plan_owners():
    """Python aliases and fixed plan arrays must be inventoried independently from physical allocations."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_descriptor_inventory")

    class Owner:
        def __init__(self, ptr, nbytes):
            self.warp = type("Warp", (), {"ptr": ptr, "device": "cuda:0", "nbytes": nbytes})()

    first_alias = Owner(3, 24)
    second_alias = Owner(3, 24)
    joint_indices = Owner(5, 16)
    fixed_input = Owner(7, 32)
    store = type("Store", (), {"_fields": {"stiffness": first_alias, "damping": second_alias}})()
    execution_range = type(
        "Range",
        (),
        {"joint_indices": joint_indices, "gather_inputs": (fixed_input,), "staging": {}},
    )()
    plan = type(
        "Plan",
        (),
        {"stateless_ranges": (execution_range,), "eager_segments": (), "static_scatter_epochs": ()},
    )()
    binding = type("Binding", (), {"execution_plan": plan, "backend_parameter_staging": None})()
    generation = type(
        "Generation",
        (),
        {
            "stores": {object: store},
            "joint_store": type("Joint", (), {"_fields": {}, "_compatibility_projections": {}})(),
            "bindings": (binding,),
        },
    )()

    report = benchmark._GlobalIntrospector().inspect(generation)
    assert report["canonical_allocation_count"] == 1
    assert report["storage_wrapper_count"] == 4
    assert report["plan_staging_owner_count"] == 2
    assert report["plan_staging_owner_bytes"] == 48


def test_global_introspection_counts_exact_python_descriptor_categories():
    """The world-scaling descriptor metric must count manager structures, not tensor wrappers."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_python_descriptors")
    registration = object()
    group_binding = object()
    group = type("Group", (), {})()
    group._parameter_binding = group_binding
    store = object()
    execution_range = object()
    eager_segment = object()
    plan = type("Plan", (), {"stateless_ranges": (execution_range,), "eager_segments": (eager_segment,)})()
    articulation_binding = type("ArticulationBinding", (), {"registration": registration, "execution_plan": plan})()
    view = object()
    manager = type("Manager", (), {"_registrations": [registration], "_views": {"robot": view}})()
    generation = type(
        "Generation",
        (),
        {
            "stores": {object: store},
            "joint_store": type("Joint", (), {"_fields": {}, "_compatibility_projections": {}})(),
            "groups": {"robot": {"group": group}},
            "bindings": (articulation_binding,),
        },
    )()

    report = benchmark._GlobalIntrospector().inspect(generation, manager=manager)
    assert report["python_descriptor_count"] == 9
    assert report["python_descriptor_counts"] == {
        "registration": 1,
        "resolved_group": 1,
        "binding": 1,
        "store": 1,
        "articulation_binding": 1,
        "view": 1,
        "execution_plan": 1,
        "execution_range": 1,
        "eager_segment": 1,
    }


def test_global_introspection_counts_aggregate_executor_binding():
    """A copied stateless range executor owns a distinct group binding and must not disappear from the inventory."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_executor_descriptor")
    facade_binding = object()
    executor_binding = object()
    group = type("Group", (), {})()
    group._parameter_binding = facade_binding
    executor = type("Executor", (), {})()
    executor._parameter_binding = executor_binding
    execution_range = type("Range", (), {"executor": executor})()
    plan = type("Plan", (), {"stateless_ranges": (execution_range,), "eager_segments": ()})()
    articulation_binding = type("ArticulationBinding", (), {"registration": object(), "execution_plan": plan})()
    generation = type(
        "Generation",
        (),
        {
            "stores": {object: object()},
            "joint_store": type("Joint", (), {"_fields": {}, "_compatibility_projections": {}})(),
            "groups": {"robot": {"group": group}},
            "bindings": (articulation_binding,),
        },
    )()
    report = benchmark._GlobalIntrospector().inspect(generation)
    assert report["python_descriptor_counts"]["binding"] == 2
    assert report["python_descriptor_count"] == 8


def test_current_and_develop_structural_values_are_null():
    """Only global observations expose actual structural ownership values."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_structure")
    assert benchmark._MemoryAdapter("develop", "cpu").introspect() is None
    assert benchmark._MemoryAdapter("current", "cpu").introspect() is None


def test_scoped_instrumentation_restores_wrapped_sites():
    """Observation wrappers must not leak outside the measured boundary."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_instrument")

    class Warp:
        def launch(self):
            pass

        def launch_tiled(self):
            pass

        def copy(self):
            pass

    warp = Warp()
    launch = warp.launch
    with benchmark._ScopedInstrumentation(warp):
        warp.launch()
    assert warp.launch.__func__ is launch.__func__


def test_scoped_instrumentation_counts_transfer_direction_and_graph_replay():
    """Removing copy-direction or graph-replay observation must change the reported counters."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_transfer_instrument")

    class Array:
        def __init__(self, device, nbytes):
            self.device = device
            self.nbytes = nbytes

    class Warp:
        def launch(self, *args, **kwargs):
            del args, kwargs

        def launch_tiled(self, *args, **kwargs):
            del args, kwargs

        def copy(self, dest, src, count=None):
            del dest, src, count

        def capture_launch(self, graph, stream=None):
            del graph, stream

    warp = Warp()
    with benchmark._ScopedInstrumentation(warp) as counter:
        warp.copy(Array("cuda:0", 16), Array("cpu", 16))
        warp.copy(Array("cpu", 12), Array("cuda:0", 12))
        counter.record_readback()
        warp.capture_launch(object())
    assert counter.as_record() == {
        "launches": {"copy": 2, "capture_launch": 1},
        "h2d_bytes": None,
        "d2h_sync_count": None,
        "warp_h2d_bytes": 16,
        "warp_d2h_copies": 1,
        "warp_d2h_sync_count": 1,
        "torch_transfer_observation": "unavailable",
    }


def test_torch_transfer_ledger_detects_h2d_and_readback_but_excludes_harness_sync():
    """The selected Torch observer must count transfer bytes and workload syncs, not the final timer sync."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_torch_transfer_ledger")

    class Tensor:
        def __init__(self, device, nbytes):
            self.device = device
            self.nbytes = nbytes

    ledger = benchmark._TorchTransferLedger()
    ledger.record_transfer(Tensor("cuda:0", 16), Tensor("cpu", 16), synchronizing=False)
    ledger.record_transfer(Tensor("cpu", 12), Tensor("cuda:0", 12), synchronizing=True)
    ledger.record_readback(Tensor("cuda:0", 4))
    ledger.record_readback(Tensor("cuda:0", 4), final_timing_sync=True)
    assert ledger.as_record() == {
        "observation": "torch_dispatch",
        "h2d_bytes": 16,
        "d2h_sync_count": 2,
    }


def test_final_harness_sync_is_excluded_from_d2h_count():
    """The timing synchronization is tagged separately from workload readbacks."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_sync")
    counter = benchmark._ScopedInstrumentation(None)
    counter.record_readback()
    counter.record_readback(final_timing_sync=True)
    assert counter.d2h_sync_count == 1


def test_capture_and_replay_observation_scopes_are_separate():
    """Capture hooks cannot be active during steady graph replay."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_scopes")
    adapter = benchmark._MemoryAdapter("global", "cpu")
    scopes = benchmark.observe_runtime_scopes(adapter, 1)
    assert scopes == ["capture", "replay"]


def test_runtime_measurement_reports_total_and_per_application_time():
    """Runtime evidence must distinguish process total from one application."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runtime_timing")
    adapter = benchmark._MemoryAdapter("global", "cpu")
    row = benchmark.RuntimeRow("implicit", 1, "cached_eager", "cached_eager", 1)
    result = benchmark.measure_runtime(adapter, row, warmups=2, iterations=4)
    assert result["status"] == "accepted"
    assert result["timing"]["total_ms"] >= 0.0
    assert result["timing"]["per_application_ms"] == pytest.approx(result["timing"]["total_ms"] / 4)
    assert result["timing"]["samples_ms"] == pytest.approx([result["timing"]["per_application_ms"]])
    assert result["timing"]["application_count"] == 4
    assert adapter.applications == 7
    assert result["counters"]["transfer_probe"]["h2d_bytes"] == 0
    assert result["counters"]["transfer_probe"]["d2h_sync_count"] == 0
    assert result["counters"]["transfer_probe"]["torch_transfer_observation"] == "torch_dispatch"


def test_runtime_warmup_precedes_allocator_baseline_and_replay_scope(monkeypatch):
    """Warmup launches and allocations must not contaminate the steady replay observation."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runtime_scope_order")
    events = []

    class Adapter(benchmark._MemoryAdapter):
        def run_execution(self, count):
            self.applications += count
            events.append(("run", count, self.applications))

    class Instrumentation:
        instances = 0

        def __init__(self, _warp):
            self.name = ("capture", "warmup", "replay")[self.instances]
            type(self).instances += 1

        def __enter__(self):
            events.append(("enter", self.name))
            return self

        def __exit__(self, *_args):
            events.append(("exit", self.name))

        def as_record(self):
            return {"scope": self.name}

    adapter = Adapter("global", "cpu")
    monkeypatch.setattr(benchmark, "_ScopedInstrumentation", Instrumentation)
    monkeypatch.setattr(
        benchmark,
        "_steady_allocation_bytes",
        lambda _device: events.append(("allocation", adapter.applications)) or 0,
    )
    monkeypatch.setattr(
        benchmark,
        "_time_runtime_execution",
        lambda timed_adapter, count: timed_adapter.run_execution(count) or 8.0,
    )
    monkeypatch.setattr(
        benchmark,
        "_observe_transfer_replay",
        lambda observed_adapter, _warp: observed_adapter.run_execution(1) or {"h2d_bytes": 0, "d2h_sync_count": 0},
    )
    row = benchmark.RuntimeRow("implicit", 1, "cached_eager", "cached_eager", 1)
    result = benchmark.measure_runtime(adapter, row, warmups=2, iterations=4)
    assert events == [
        ("enter", "warmup"),
        ("run", 2, 2),
        ("exit", "warmup"),
        ("run", 1, 3),
        ("allocation", 3),
        ("enter", "replay"),
        ("run", 4, 7),
        ("exit", "replay"),
        ("allocation", 7),
    ]
    assert result["counters"]["warmup"] == {"scope": "warmup"}


def test_runtime_snapshots_global_pointers_before_and_after_warmup_and_replay(monkeypatch):
    """Pointer stability must be sampled outside the timed call, not inferred from final record emission."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runtime_pointer_snapshots")
    snapshots = []

    class Adapter(benchmark._MemoryAdapter):
        def introspect(self):
            snapshots.append(self.applications)
            return {
                "pointer_replacements": 0,
                "pointer_snapshot_count": len(snapshots),
            }

    adapter = Adapter("global", "cpu")
    monkeypatch.setattr(
        benchmark,
        "_observe_transfer_replay",
        lambda observed_adapter, _warp: observed_adapter.run_execution(1) or {"h2d_bytes": 0, "d2h_sync_count": 0},
    )
    row = benchmark.RuntimeRow("implicit", 1, "cached_eager", "cached_eager", 1)
    result = benchmark.measure_runtime(adapter, row, warmups=2, iterations=4)
    assert snapshots == [0, 2, 3, 7]
    assert result["counters"]["pointer_stability"] == {
        "observation": "global_introspection",
        "pointer_replacements": 0,
        "pointer_snapshot_count": 4,
    }


@pytest.mark.parametrize(
    ("phase", "expected_warmups", "expected_samples"),
    (("cold", 0, 1), ("warm", 10, 100)),
)
def test_build_measurement_uses_fresh_workloads_and_adapters_per_sample(phase, expected_warmups, expected_samples):
    """Cold and warm construction must never reuse an adapter or workload across samples."""
    benchmark = _load(_BENCHMARK, f"actuator_benchmark_build_measurement_{phase}")
    row = benchmark.expand_build_matrix("B1")[0]
    workloads = []
    adapters = []

    class Adapter(benchmark._MemoryAdapter):
        def __init__(self, revision, device):
            super().__init__(revision, device)
            self.builds = 0
            self.firsts = 0
            self.closes = 0
            adapters.append(self)

        def build_workload(self, workload):
            self.builds += 1
            super().build_workload(workload)

        def first_application(self, workload):
            self.firsts += 1
            super().first_application(workload)

        def close(self):
            self.closes += 1
            super().close()

    def workload_factory(build_row, device):
        workload = benchmark.make_workload(build_row, device)
        workloads.append(workload)
        return workload

    result = benchmark.measure_build(
        "develop",
        row,
        "cpu",
        phase,
        adapter_factory=Adapter,
        workload_factory=workload_factory,
    )
    total = expected_warmups + expected_samples
    assert len(workloads) == len(adapters) == total
    assert len({id(workload) for workload in workloads}) == total
    assert all(adapter.builds == adapter.firsts == adapter.closes == 1 for adapter in adapters)
    assert result["timing"]["warmup_construction_count"] == expected_warmups
    assert result["timing"]["construction_count"] == expected_samples
    assert len(result["timing"]["samples_ms"]) == expected_samples


def test_build_measurement_starts_after_fixture_creation_and_closes_after_first_application(monkeypatch):
    """The external construction boundary excludes fixture creation but includes the first synchronized application."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_build_boundary")
    row = benchmark.expand_build_matrix("B1")[0]
    events = []

    class Adapter(benchmark._MemoryAdapter):
        def __init__(self, revision, device):
            events.append("adapter")
            super().__init__(revision, device)

        def build_workload(self, workload):
            events.append("build")
            super().build_workload(workload)

        def introspect(self):
            events.append("snapshot")
            return

        def first_application(self, workload):
            events.append("first")
            super().first_application(workload)

        def close(self):
            events.append("close")
            super().close()

    clock_values = iter((1_000_000, 4_000_000))
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: events.append("clock") or next(clock_values))
    monkeypatch.setattr(benchmark, "_synchronize_boundary", lambda _device: events.append("sync"))

    def workload_factory(build_row, device):
        events.append("workload")
        return benchmark.make_workload(build_row, device)

    result = benchmark.measure_build(
        "develop",
        row,
        "cpu",
        "cold",
        adapter_factory=Adapter,
        workload_factory=workload_factory,
    )
    assert events == [
        "workload",
        "adapter",
        "clock",
        "build",
        "snapshot",
        "first",
        "sync",
        "clock",
        "snapshot",
        "close",
    ]
    assert result["timing"]["samples_ms"] == [3.0]


def test_build_measurement_cleans_adapter_and_checkpoint_after_build_failure(tmp_path):
    """A failed measured construction must close its adapter and remove its driver-owned checkpoint."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_build_failure_cleanup")
    row = benchmark.expand_build_matrix("B1")[0]
    checkpoint = tmp_path / "network.pt"
    checkpoint.write_bytes(b"checkpoint")
    closes = []

    class Adapter(benchmark._MemoryAdapter):
        def build_workload(self, workload):
            self.workload = workload
            raise RuntimeError("build failed")

        def close(self):
            closes.append(True)
            super().close()

    def workload_factory(build_row, device):
        return benchmark._Workload(
            build_row,
            device,
            ("joint_0",),
            ((1.0,),),
            (0.1,),
            str(checkpoint),
        )

    with pytest.raises(RuntimeError, match="build failed"):
        benchmark.measure_build(
            "develop",
            row,
            "cpu",
            "cold",
            adapter_factory=Adapter,
            workload_factory=workload_factory,
        )
    assert closes == [True]
    assert not checkpoint.exists()


def test_build_measurement_keeps_global_decomposition_out_of_comparable_timing():
    """Registration/finalization diagnostics are nested evidence, not extra comparable samples."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_build_decomposition")
    row = benchmark.expand_build_matrix("B1")[0]

    class Adapter(benchmark._MemoryAdapter):
        def build_workload(self, workload):
            super().build_workload(workload)
            self.build_decomposition_ms = {"registration": 1.25, "finalization": 2.5}

    result = benchmark.measure_build(
        "global",
        row,
        "cpu",
        "cold",
        adapter_factory=Adapter,
    )
    assert len(result["timing"]["samples_ms"]) == 1
    assert "registration" not in result["timing"]
    assert result["counters"]["global_decomposition_samples_ms"] == {
        "registration": [1.25],
        "finalization": [2.5],
    }


def test_build_coordinator_uses_one_fresh_child_per_cold_row(tmp_path):
    """Cold construction samples are isolated at the process boundary."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_coordinator")
    calls = []
    coordinator = benchmark.Coordinator(tmp_path, runner=lambda command: calls.append(command) or {"returncode": 0})
    coordinator.schedule_cold_children([benchmark.expand_build_matrix("B1")[0]], repetitions=2)
    assert len(calls) == 2 and all("--child_row" in call for call in calls)


def test_coordinator_never_constructs_a_collection(tmp_path, monkeypatch):
    """The parent only launches children and never imports target actuator APIs."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_parent")
    monkeypatch.setattr(benchmark, "select_adapter", lambda *_args, **_kwargs: pytest.fail("parent imported adapter"))
    benchmark.Coordinator(tmp_path, runner=lambda _: {"returncode": 0}).schedule_cold_children([], 1)


def test_harness_copy_hash_is_immutable(tmp_path):
    """A resumed run rejects a candidate driver whose copied bytes changed."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_harness")
    source = tmp_path / "driver.py"
    source.write_text("x = 1\n")
    copied, digest = benchmark.prepare_harness(tmp_path / "run", source)
    assert copied.stat().st_mode & 0o222 == 0
    assert benchmark.prepare_harness(tmp_path / "run", source)[1] == digest
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="digest"):
        benchmark.prepare_harness(tmp_path / "run", source)


def test_attempt_allocator_is_atomic_and_never_overwrites(tmp_path):
    """Attempt documents get an exclusive directory and immutable final JSON."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_attempts")
    first = benchmark.allocate_attempt_dir(tmp_path)
    second = benchmark.allocate_attempt_dir(tmp_path)
    assert first.name == "attempt-01" and second.name == "attempt-02"
    benchmark.write_attempt_atomically(first, _record(benchmark))
    with pytest.raises(FileExistsError):
        benchmark.write_attempt_atomically(first, _record(benchmark))


def test_rejected_pair_is_retained_and_retry_uses_next_attempt(tmp_path):
    """Retries add evidence instead of changing rejected documents."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_retry")
    first = benchmark.allocate_attempt_dir(tmp_path)
    benchmark.write_attempt_atomically(first, _record(benchmark, status="rejected"))
    assert benchmark.allocate_attempt_dir(tmp_path).name == "attempt-02"
    assert json.loads((first / "attempt.json").read_text())["status"] == "rejected"


def test_supported_comparison_has_six_balanced_pair_ids():
    """Pair scheduling counterbalances baseline-first and global-first order."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_pairs")
    pairs = benchmark.balanced_pair_schedule("develop", "global")
    assert [pair[0] for pair in pairs] == [f"{n:02}" for n in range(1, 7)]
    assert [pair[1] for pair in pairs] == ["D-G"] * 3 + ["G-D"] * 3


def test_missing_required_gpu_telemetry_rejects_final_acceptance():
    """GPU acceptance requires complete pre/post telemetry evidence."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_telemetry")
    assert benchmark.validate_pair_telemetry([], [], device="cuda:0") == ["required telemetry unavailable"]
    assert benchmark.validate_pair_telemetry([], [], device="cpu") == []


def test_summary_uses_paired_process_medians_and_seed_42(tmp_path):
    """Summary ratios are based on six paired process medians, not inner samples."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_summary_driver")
    summary = _load(_SUMMARY, "actuator_benchmark_summary")
    records = []
    for number in range(1, 7):
        item = _record(benchmark)
        item["identity"]["observation_key"] = f"B1/pair-{number:02}"
        item["pair_order"] = "D-G" if number <= 3 else "G-D"
        item["members"][0]["timing"] = {"samples_ms": [float(number), 1000.0]}
        item["members"][1]["timing"] = {"samples_ms": [float(number * 2), 2000.0]}
        records.append(item)
    report = summary.summarize_records(records, "c" * 40, 42)
    assert report["comparisons"][0]["ratio_median"] == pytest.approx(2.0)
    assert report["bootstrap_seed"] == 42


def test_summary_rejects_duplicate_or_missing_pair_ids():
    """A comparison needs exactly the six distinct scheduled pair identifiers."""
    summary = _load(_SUMMARY, "actuator_benchmark_summary_pairs")
    with pytest.raises(ValueError, match="pair"):
        summary.validate_pair_ids(["01", "01"])


def test_summary_rejects_unbalanced_order_or_wrong_member():
    """A pair must contain its declared baseline and global members in scheduled order."""
    summary = _load(_SUMMARY, "actuator_benchmark_summary_members")
    with pytest.raises(ValueError, match="order"):
        summary.validate_orders(["D-G"] * 6)


def test_summary_rejects_mixed_sha_or_harness_identity():
    """Accepted statistics require one candidate and harness identity."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_summary_identity_driver")
    summary = _load(_SUMMARY, "actuator_benchmark_summary_identity")
    first = _record(benchmark)
    second = _record(benchmark)
    second["identity"]["attempt_id"] = "attempt-02"
    second["identity"]["harness_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="harness"):
        summary.validate_records([first, second], "c" * 40)


def test_summary_rejects_unsupported_timing_and_missing_capability():
    """Summary validation repeats the driver invariant before statistics."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_summary_valid_driver")
    summary = _load(_SUMMARY, "actuator_benchmark_summary_valid")
    record = _record(benchmark)
    del record["members"][0]["capability"]
    with pytest.raises(ValueError, match="capability"):
        summary.validate_records([record], "c" * 40)


def test_summary_keeps_immutable_rejections_out_of_accepted_statistics():
    """Rejected attempts remain report evidence but never become numeric samples."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_summary_rejected_driver")
    summary = _load(_SUMMARY, "actuator_benchmark_summary_rejected")
    accepted = _record(benchmark)
    rejected = _record(benchmark, status="rejected")
    rejected["identity"]["attempt_id"] = "attempt-02"
    report = summary.summarize_records([accepted, rejected], "c" * 40, 42)
    assert report["accepted_attempt_count"] == 1 and report["rejected_attempt_count"] == 1


def test_summary_writes_json_csv_and_markdown_from_validated_report(tmp_path):
    """The strict reader emits all three requested report representations."""
    summary = _load(_SUMMARY, "actuator_benchmark_summary_outputs")
    report = {
        "candidate_sha": "c" * 40,
        "comparisons": [
            {
                "observation_key": "B1",
                "accepted_pair_count": 6,
                "ratio_median": 2.0,
                "ratio_mean": 2.0,
                "ratio_p95": 2.0,
                "ratio_dispersion": 0.0,
                "ratio_bootstrap_95": [2.0, 2.0],
            }
        ],
    }
    summary.write_outputs(report, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "benchmark-summary.json",
        "benchmark-summary.csv",
        "benchmark-summary.md",
    }


def test_summary_rejects_incomplete_six_pair_manifest():
    """Accepted pair statistics reject a five-pair manifest before calculation."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_manifest_driver")
    summary = _load(_SUMMARY, "actuator_benchmark_manifest")
    records = []
    for number in range(1, 6):
        item = _record(benchmark)
        item["identity"]["observation_key"] = f"B1/pair-{number:02}"
        item["identity"]["attempt_id"] = f"attempt-{number:02}"
        item["pair_order"] = "D-G" if number <= 3 else "G-D"
        records.append(item)
    with pytest.raises(ValueError, match="pair IDs"):
        summary.summarize_records(records, "c" * 40, 42)


def test_gpu_telemetry_requires_exactly_twenty_pre_and_post_samples():
    """A final GPU pair records exactly forty cadence samples before acceptance."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_telemetry_count")
    sample = benchmark.TelemetrySample(0.0, 40.0, 0.0, 1000.0, 1000.0, None, ())
    assert benchmark.validate_pair_telemetry([sample] * 19, [sample] * 20, "cuda:0") == [
        "required telemetry unavailable"
    ]
    assert benchmark.validate_pair_telemetry([sample] * 20, [sample] * 20, "cuda:0") == []


def test_workload_preserves_each_requested_group_as_a_distinct_joint_domain():
    """A B5/12 workload must not silently collapse the twelve groups to three joints."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_group_domains")
    row = next(
        candidate
        for candidate in benchmark.expand_build_matrix("B5")
        if candidate.actuator_types == ("ideal_pd",) and candidate.groups == 12
    )
    workload = benchmark.make_workload(row, "cpu")
    assert workload.joint_names == tuple(f"joint_{index}" for index in range(12))
    assert len(workload.group_values) == 12
    assert [values[0] for values in workload.group_values] == [float(index + 1) for index in range(12)]


def test_global_introspection_reads_real_generation_owners_not_dictionary_keys():
    """Canonical allocation data must come from stores, plans, staging, and joint storage."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_real_introspection")

    class Owner:
        def __init__(self, ptr, nbytes):
            self.warp = type("Warp", (), {"ptr": ptr, "device": "cuda:0", "nbytes": nbytes})()

    canonical = Owner(3, 24)
    staging = Owner(5, 40)
    store = type("Store", (), {"_fields": {"stiffness": canonical}})()
    plan = type("Plan", (), {"_staging": {"implicit": staging}})()
    binding = type("Binding", (), {"execution_plan": plan, "backend_parameter_staging": staging})()
    generation = type(
        "Generation",
        (),
        {"stores": {object: store}, "joint_store": type("Joint", (), {"_fields": {}})(), "bindings": (binding,)},
    )()

    report = benchmark._GlobalIntrospector().inspect(generation)
    assert report["canonical_allocation_count"] == 1
    assert report["canonical_allocation_bytes"] == 24
    assert report["plan_staging_owner_count"] == 1
    assert report["plan_staging_owner_bytes"] == 40


def test_global_introspection_deduplicates_domains_and_observes_projection_lifecycle():
    """Aliases, replacement, projection allocation, and cleared ownership must be measured from owners."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_introspection_lifecycle")

    class Owner:
        def __init__(self, ptr, nbytes):
            self.warp = type("Warp", (), {"ptr": ptr, "device": "cuda:0", "nbytes": nbytes})()

    canonical = Owner(3, 24)
    projection = Owner(5, 40)
    store = type("Store", (), {"_fields": {"stiffness": canonical}})()
    staging = type("Staging", (), {"_targets": {"stiffness": canonical}})()
    joint_store = type(
        "Joint",
        (),
        {"_fields": {}, "_compatibility_projections": {(1, "soft_joint_vel_limits"): projection}},
    )()
    binding = type("Binding", (), {"execution_plan": None, "backend_parameter_staging": staging})()
    generation = type(
        "Generation", (), {"stores": {object: store}, "joint_store": joint_store, "bindings": (binding,)}
    )()

    introspector = benchmark._GlobalIntrospector()
    first = introspector.inspect(generation)
    assert first["canonical_allocation_count"] == 1
    assert first["plan_staging_owner_count"] == 0
    assert first["projection_bytes"] == 40
    assert first["pointer_replacements"] == 0

    joint_store._compatibility_projections[(1, "soft_joint_vel_limits")] = Owner(7, 40)
    replaced = introspector.inspect(generation)
    assert replaced["pointer_replacements"] == 1
    assert replaced["pointer_snapshot_count"] == 2

    cleared = introspector.inspect(None)
    assert cleared["canonical_allocation_count"] == 0
    assert cleared["plan_staging_owner_count"] == 0
    assert cleared["projection_bytes"] == 0
    assert cleared["clear_state_ownership"] == 0
    assert cleared["pointer_snapshot_count"] == 0


def test_global_b0_b2_b6_and_b8_probes_exercise_manager_lifecycle_and_projections():
    """Global-only rows must use manager finalization, lazy projections, clear, and re-registration."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_global_lifecycle")
    results = {
        case: benchmark.run_global_structural_case(
            benchmark.make_workload(benchmark.expand_build_matrix(case)[0], "cpu")
        )
        for case in ("B0", "B2", "B6", "B8")
    }
    assert results["B0"]["cleared"] is True
    assert results["B2"]["articulation_count"] == 2
    assert results["B2"]["applied_facades"] == ("benchmark-0", "benchmark-1")
    assert results["B2"]["submission_counts"] == (1, 1)
    assert results["B6"]["projection_states"] == ("untouched", "first", "repeated", "both")
    assert results["B6"]["projection_count"] == 2
    assert results["B6"]["projection_launches"] > results["B6"]["projection_count"]
    assert results["B6"]["projection_bytes_by_state"]["untouched"] == 0
    assert results["B6"]["projection_bytes_by_state"]["first"] > 0
    assert results["B6"]["projection_bytes_by_state"]["repeated"] == results["B6"]["projection_bytes_by_state"]["first"]
    assert results["B6"]["projection_bytes_by_state"]["both"] > results["B6"]["projection_bytes_by_state"]["repeated"]
    assert results["B6"]["repeated_pointer_stable"] is True
    assert results["B6"]["repeat_retained_owner_delta_bytes"] == 0
    assert results["B6"]["repeat_allocator_observation"] == "unavailable_cpu"
    assert results["B6"]["repeat_allocator_delta_bytes"] is None
    assert results["B6"]["repeat_allocator_peak_delta_bytes"] is None
    assert results["B6"]["repeat_allocation_free"] is None
    assert results["B8"]["re_registered"] is True
    assert results["B8"]["same_manager"] is True
    assert results["B8"]["old_view_stale"] is True
    assert results["B8"]["clear_state_ownership"] == 0


def test_global_adapter_reports_only_current_generation_structural_owners():
    """A real global B1 adapter must expose concrete current-generation structural ownership."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_live_introspection")
    adapter = benchmark._GlobalCollectionAdapter("global", "cpu")
    workload = benchmark.make_workload(benchmark.expand_build_matrix("B1")[0], "cpu")
    try:
        adapter.build_workload(workload)
        report = adapter.introspect()
        assert report is not None
        assert report["canonical_allocation_count"] > 0
        assert report["storage_wrapper_count"] > 0
        assert report["python_descriptor_count"] > 0
        assert report["plan_staging_owner_count"] > 0
    finally:
        adapter.close()


def test_global_b1_python_descriptor_count_does_not_scale_with_worlds():
    """The exact Python descriptor inventory must be independent of a B1 articulation's world count."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_live_descriptor_scaling")
    reports = []
    for worlds in (1, 64):
        base = benchmark.expand_build_matrix("B1")[0]
        row = benchmark.BuildRow(
            base.case,
            worlds,
            base.num_sources,
            base.num_articulations,
            base.groups,
            base.actuator_types,
            base.global_only,
        )
        adapter = benchmark._GlobalCollectionAdapter("global", "cpu")
        workload = benchmark.make_workload(row, "cpu")
        try:
            adapter.build_workload(workload)
            reports.append(adapter.introspect())
        finally:
            adapter.close()
    assert reports[0]["python_descriptor_count"] == reports[1]["python_descriptor_count"]
    assert reports[0]["python_descriptor_counts"] == reports[1]["python_descriptor_counts"]
    assert reports[0]["python_descriptor_counts"]["binding"] == 4


def test_cold_build_cli_publishes_one_measured_construction_and_pointer_snapshots(tmp_path):
    """A cold child must route through the measured boundary and retain its post-build stability evidence."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_build_cli_cold")
    output = tmp_path / "cold"
    assert (
        benchmark.main(
            [
                "--mode",
                "build",
                "--revision",
                "global",
                "--case",
                "B1",
                "--num_worlds",
                "1",
                "--phase",
                "cold",
                "--device",
                "cpu",
                "--output_path",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads(next(output.glob("*/*/attempt.json")).read_text())
    member = record["members"][0]
    assert member["timing"]["construction_count"] == 1
    assert len(member["timing"]["samples_ms"]) == 1
    assert set(member["counters"]["global_decomposition_samples_ms"]) == {
        "manager_construction",
        "control_construction",
        "registration",
        "finalization",
        "state_reset",
    }
    assert member["structural"]["pointer_snapshot_count"] >= 2
    assert member["structural"]["pointer_replacements"] == 0


@pytest.mark.parametrize("case", ("B0", "B2", "B6", "B8"))
def test_global_structural_cli_records_the_case_specific_observation(case, tmp_path):
    """The executable child must publish each global-only structural sequence instead of a generic build."""
    benchmark = _load(_BENCHMARK, f"actuator_benchmark_structural_cli_{case}")
    output = tmp_path / case
    assert (
        benchmark.main(
            [
                "--mode",
                "build",
                "--revision",
                "global",
                "--case",
                case,
                "--num_worlds",
                "1",
                "--device",
                "cpu",
                "--benchmark_formatter",
                "schema",
                "--output_path",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads(next(output.glob("*/*/attempt.json")).read_text())
    benchmark.validate_attempt(record)
    structural = record["members"][0]["structural"]
    if case == "B0":
        assert record["boundary"] == "empty_finalize_clear"
        assert structural["cleared"] is True
    elif case == "B2":
        assert structural["articulation_count"] == 2
        assert structural["submission_counts"] == [1, 1]
        assert structural["applied_facades"] == ["benchmark-0", "benchmark-1"]
    elif case == "B6":
        assert structural["projection_launches"] > structural["projection_count"] == 2
        assert structural["repeat_retained_owner_delta_bytes"] == 0
        assert structural["repeat_allocator_observation"] == "unavailable_cpu"
        assert structural["repeat_allocation_free"] is None
    else:
        assert structural["same_manager"] is True
        assert structural["old_view_stale"] is True
        assert structural["clear_state_ownership"] == 0


def test_global_only_cli_rejects_historical_revision_before_adapter_selection(tmp_path, monkeypatch):
    """A global-only row must be rejected before any historical actuator construction is selected."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_structural_revision_gate")
    monkeypatch.setattr(benchmark, "select_adapter", lambda *_args: pytest.fail("adapter selection was reached"))
    with pytest.raises(RuntimeError, match="global-only"):
        benchmark.main(
            [
                "--mode",
                "build",
                "--revision",
                "develop",
                "--case",
                "B2",
                "--num_worlds",
                "1",
                "--device",
                "cpu",
                "--output_path",
                str(tmp_path),
            ]
        )


def test_global_adapter_close_is_idempotent_permanent_and_rejects_registration():
    """Adapter teardown must call permanent manager close and remain safe when repeated."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_manager_close")
    adapter = benchmark._GlobalCollectionAdapter("global", "cpu")
    workload = benchmark.make_workload(benchmark.expand_build_matrix("B1")[0], "cpu")
    adapter.build_workload(workload)
    manager = adapter.manager
    adapter.close()
    adapter.close()
    assert manager._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        manager.register_articulation(
            key="late",
            cfgs={},
            control=None,
            replication_cfg_id=1,
            debug_validation=False,
            debug_value_resolution=False,
        )


def test_runtime_cli_executes_requested_child_and_writes_runtime_evidence(tmp_path):
    """Runtime mode must execute the selected workload rather than return without an attempt."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runtime_cli")
    output = tmp_path / "runtime"
    assert (
        benchmark.main(
            [
                "--mode",
                "runtime",
                "--revision",
                "global",
                "--child_row",
                "implicit:g1:cached_eager",
                "--num_worlds",
                "1",
                "--warmup_iterations",
                "1",
                "--num_iterations",
                "2",
                "--device",
                "cpu",
                "--benchmark_formatter",
                "schema",
                "--output_path",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads(next(output.glob("*/*/attempt.json")).read_text())
    benchmark.validate_attempt(record)
    member = record["members"][0]
    assert record["boundary"] == "runtime_application"
    assert member["resolved_row"]["num_worlds"] == 1
    assert member["timing"]["application_count"] == 2
    assert member["timing"]["per_application_ms"] == pytest.approx(member["timing"]["total_ms"] / 2)


def test_b7_builds_deterministic_local_neural_and_eager_fallback_groups():
    """B7 must create neural, delayed, remotized, and opaque groups without a download."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_b7")
    row = benchmark.expand_build_matrix("B7")[0]
    adapter = benchmark._GlobalCollectionAdapter("global", "cpu")
    workload = benchmark.make_workload(row, "cpu")
    try:
        adapter.build_workload(workload)
        adapter.first_application(workload)
        assert set(adapter.view) == {"group_0", "group_1", "group_2", "group_3"}
        neural = adapter.view["group_0"]
        delayed = adapter.view["group_1"]
        remotized = adapter.view["group_2"]
        opaque = adapter.view["group_3"]
        assert neural.computed_effort[0, 0].item() == pytest.approx(0.2)
        assert delayed.positions_delay_buffer.time_lags.unique().tolist() == [1]
        assert remotized.positions_delay_buffer.time_lags.unique().tolist() == [1]
        assert delayed.applied_effort[0, 0].item() == pytest.approx(0.3)
        assert remotized.applied_effort[0, 0].item() == pytest.approx(0.4)
        adapter.view.command.position.torch.fill_(0.2)
        adapter.run_execution(1)
        assert neural.computed_effort[0, 0].item() == pytest.approx(0.4)
        assert delayed.applied_effort[0, 0].item() == pytest.approx(0.3)
        assert remotized.applied_effort[0, 0].item() == pytest.approx(0.4)
        adapter.run_execution(1)
        assert neural.computed_effort[0, 0].item() == pytest.approx(0.4)
        assert delayed.applied_effort[0, 0].item() == pytest.approx(0.6)
        assert remotized.applied_effort[0, 0].item() == pytest.approx(0.8)
        adapter.view.command.position.torch.fill_(100.0)
        adapter.run_execution(2)
        assert remotized.applied_effort[0, 0].item() == pytest.approx(20.0)
        assert type(opaque) is benchmark._opaque_actuator_type()
        plan = adapter.view._execution_plan
        assert plan is not None and not plan.stateless_ranges and len(plan.eager_segments) == 4
    finally:
        adapter.close()


def test_tiny_checkpoint_removes_partial_file_when_torchscript_creation_fails(tmp_path, monkeypatch):
    """A TorchScript failure must not leave the pre-created checkpoint path behind."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_checkpoint_cleanup")
    import torch

    original_mkstemp = benchmark.tempfile.mkstemp

    def make_partial(*args, **kwargs):
        del args, kwargs
        return original_mkstemp(dir=tmp_path, prefix="partial-", suffix=".pt")

    monkeypatch.setattr(benchmark.tempfile, "mkstemp", make_partial)
    monkeypatch.setattr(torch.jit, "trace", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("trace")))
    with pytest.raises(RuntimeError, match="trace"):
        benchmark._tiny_mlp_checkpoint("cpu")
    assert list(tmp_path.iterdir()) == []
