# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the private actuator-collection benchmark driver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _coordinate_argv(tmp_path, *, matrix="runtime", device="cpu", batch_id="runtime-01", pair_repetitions=2):
    develop_sha = "d" * 40
    current_sha = "c" * 40
    global_sha = "a" * 40
    return [
        "--mode",
        "coordinate",
        "--matrix",
        matrix,
        "--develop_worktree",
        str(tmp_path / "develop"),
        "--develop_sha",
        develop_sha,
        "--current_worktree",
        str(tmp_path / "current"),
        "--current_sha",
        current_sha,
        "--global_worktree",
        str(tmp_path / "global"),
        "--global_sha",
        global_sha,
        "--candidate_sha",
        global_sha,
        "--run_root",
        str(tmp_path / "run"),
        "--batch_id",
        batch_id,
        "--cold_repetitions",
        "2",
        "--pair_repetitions",
        str(pair_repetitions),
        "--warmup_iterations",
        "1",
        "--num_iterations",
        "1",
        "--device",
        device,
        "--benchmark_formatter",
        "schema",
    ]


class _FakeChildRunner:
    """Write exact child-member results at the subprocess boundary."""

    def __init__(self, benchmark, *, fail_calls=()):
        self.benchmark = benchmark
        self.fail_calls = set(fail_calls)
        self.calls = []

    def __call__(self, command, *, cwd, env):
        self.calls.append((command, cwd, env))
        call_number = len(self.calls)
        if call_number in self.fail_calls:
            return {"returncode": 17, "stdout": "partial output", "stderr": "child exploded"}

        def value(flag):
            return command[command.index(flag) + 1]

        output = Path(value("--output_path"))
        output.mkdir(parents=True, exist_ok=True)
        revision = value("--revision")
        phase = value("--phase")
        child_row = json.loads(value("--child_row"))
        requested = child_row.get("requested_execution", phase)
        effective = child_row.get("effective_execution", requested)
        payload = {
            "schema": "actuator_collection_member/v1",
            "identity": {
                "batch_id": value("--batch_id"),
                "observation_key": value("--observation_key"),
                "attempt_id": value("--attempt_id"),
                "candidate_sha": value("--candidate_sha"),
                "harness_sha256": value("--harness_sha256"),
            },
            "revision": revision,
            "revision_sha": value("--revision_sha"),
            "matrix": value("--mode"),
            "phase": phase,
            "child_row": child_row,
            "status": "accepted",
            "member": {
                "revision": revision,
                "requested_execution": requested,
                "effective_execution": effective,
                "revision_sha": value("--revision_sha"),
                "adapter": f"{revision}-adapter",
                "resolved_row": child_row,
                "source_emulation": revision != "global" and child_row.get("case") == "B3",
                "capability": {"supported": True, "reason": None},
                "timing": None if phase == "compile_prewarm" else {"samples_ms": [1.0]},
                "counters": {},
                "structural": {} if revision == "global" else None,
                "execution": {"graph_capture_live": requested == "graph"},
            },
        }
        (output / "member.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"returncode": 0, "stdout": f"ran {revision}", "stderr": ""}


def _coordinate_context(benchmark, tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text("# immutable harness\n", encoding="utf-8")
    worktrees = {revision: tmp_path / revision for revision in ("develop", "current", "global")}
    for path in worktrees.values():
        path.mkdir()
        (path / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (path / "uv.lock").write_text(f"lock-{path.name}\n", encoding="utf-8")
    return benchmark._CoordinateContext(
        batch_id="runtime-01",
        candidate_sha="a" * 40,
        revision_shas={"develop": "d" * 40, "current": "c" * 40, "global": "a" * 40},
        worktrees=worktrees,
        harness=harness.resolve(),
        harness_sha256="a" * 64,
        device="cpu",
        warmup_iterations=1,
        num_iterations=1,
        command=("coordinate",),
        initial_metadata={"python": "test-python", "platform": "test-platform", "gpu": [], "versions": {}},
        worktree_states={
            revision: benchmark.WorktreeState(sha, False)
            for revision, sha in {"develop": "d" * 40, "current": "c" * 40, "global": "a" * 40}.items()
        },
        lockfile_sha256={"develop": "d" * 64, "current": "c" * 64, "global": "a" * 64},
        benchmark_config_sha256="b" * 64,
        benchmark_config={"matrix": "runtime"},
    )


def _clean_probe(benchmark, context):
    return lambda path: benchmark.WorktreeState(context.revision_shas[path.name], False)


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


def test_final_harness_sync_is_excluded_from_d2h_count():
    """The timing synchronization is tagged separately from workload readbacks."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_sync")
    counter = benchmark._ScopedInstrumentation(None)
    counter.record_readback()
    counter.record_readback(final_timing_sync=True)
    assert counter.d2h_readbacks == 1


def test_capture_and_replay_observation_scopes_are_separate():
    """Capture hooks cannot be active during steady graph replay."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_scopes")
    adapter = benchmark._MemoryAdapter("global", "cpu")
    scopes = benchmark.observe_runtime_scopes(adapter, 1)
    assert scopes == ["capture", "replay"]


def test_coordinate_cli_requires_exact_three_revision_identity(tmp_path, capsys):
    """Final coordination cannot start without every pinned worktree and exact SHA."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_coordinate_cli")
    args = benchmark.parse_args(_coordinate_argv(tmp_path))
    assert args.develop_worktree == tmp_path / "develop"
    assert args.current_worktree == tmp_path / "current"
    assert args.global_worktree == tmp_path / "global"
    assert args.cold_repetitions == 2 and args.pair_repetitions == 2
    incomplete = _coordinate_argv(tmp_path)
    del incomplete[incomplete.index("--current_sha") : incomplete.index("--current_sha") + 2]
    with pytest.raises(SystemExit):
        benchmark.parse_args(incomplete)
    assert "--current_sha" in capsys.readouterr().err
    mismatch = _coordinate_argv(tmp_path)
    mismatch[mismatch.index("--candidate_sha") + 1] = "b" * 40
    with pytest.raises(SystemExit):
        benchmark.parse_args(mismatch)
    assert "--global_sha" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        benchmark.parse_args([*_coordinate_argv(tmp_path), "--revision", "global"])
    assert "child-only" in capsys.readouterr().err


@pytest.mark.parametrize(
    "ignored_flags",
    [
        ("--case", "B1"),
        ("--case", "B1", "--num_worlds", "2"),
        ("--case", "B1", "--num_sources", "2"),
        ("--case", "B1", "--num_articulations", "2"),
        ("--case", "B1", "--groups", "2"),
        ("--case", "B1", "--actuator_types", "implicit"),
        ("--output_path", "/tmp/ignored-coordinate-output"),
        ("--case=B1",),
        ("--output_path=/tmp/ignored-coordinate-output",),
    ],
)
def test_coordinate_cli_rejects_every_ignored_workload_or_output_flag(tmp_path, capsys, ignored_flags):
    """Coordinate mode cannot silently ignore a user-provided workload selector."""
    benchmark = _load(_BENCHMARK, f"actuator_benchmark_coordinate_ignored_{len(ignored_flags)}")
    with pytest.raises(SystemExit):
        benchmark.parse_args([*_coordinate_argv(tmp_path), *ignored_flags])
    assert "workload, selector, or output" in capsys.readouterr().err


def test_build_schedule_pairs_supported_rows_and_keeps_global_only_singletons():
    """Global-only rows never acquire manufactured historical pair members."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_build_schedule")
    observations = benchmark.build_coordinate_schedule(cold_repetitions=6, pair_repetitions=6)
    b1_cold = [
        item
        for item in observations
        if item.row_key == "B1:w1:s1:a1:g3:implicit" and item.phase == "cold" and item.comparison == "develop-global"
    ]
    assert [item.pair_id for item in b1_cold] == [f"{number:02}" for number in range(1, 7)]
    assert [item.revisions for item in b1_cold] == [("develop", "global")] * 3 + [("global", "develop")] * 3
    for case in ("B2", "B6", "B8"):
        rows = [item for item in observations if item.row_key.startswith(case + ":")]
        assert rows and all(item.kind == "singleton" and item.revisions == ("global",) for item in rows)


def test_runtime_schedule_has_supported_pairs_and_standalone_historical_graph_evidence():
    """Historical graph requests remain unsupported singletons rather than fake pairs."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runtime_schedule")
    observations = benchmark.runtime_coordinate_schedule(pair_repetitions=6)
    logical = [item for item in observations if item.row_key.startswith("implicit:g1:")]
    pairs = [item for item in logical if item.kind == "pair"]
    unsupported = [item for item in logical if item.kind == "singleton"]
    assert len(pairs) == 24
    assert {item.mode_pair for item in pairs} == {
        "current-cached_eager__global-graph",
        "current-cached_eager__global-cached_eager",
        "develop-cached_eager__global-graph",
        "develop-cached_eager__global-cached_eager",
    }
    assert {(item.revisions[0], item.requested_executions[0]) for item in unsupported} == {
        ("develop", "graph"),
        ("current", "graph"),
    }
    assert len([item for item in observations if item.kind == "pair"]) == 216
    assert len([item for item in observations if item.kind == "singleton"]) == 18


def test_build_schedule_has_complete_frozen_pair_and_singleton_counts():
    """Dropping a phase or structural row must make the final matrix incomplete."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_build_schedule_count")
    observations = benchmark.build_coordinate_schedule(6, 6)
    assert len([item for item in observations if item.kind == "pair"]) == 360
    assert len([item for item in observations if item.kind == "singleton"]) == 21


def test_pair_attempt_atomically_owns_ordered_fresh_members(tmp_path):
    """Moving member allocation outside the pair attempt would split its identity."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_atomic_pair")
    context = _coordinate_context(benchmark, tmp_path)
    runner = _FakeChildRunner(benchmark)
    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=runner, worktree_probe=_clean_probe(benchmark, context), sleep=lambda _: None
    )
    observation = next(
        item
        for item in benchmark.runtime_coordinate_schedule(2)
        if item.mode_pair == "develop-cached_eager__global-graph" and item.pair_id == "01"
    )
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert len(list(attempt.parent.glob("attempt-*"))) == 1
    assert [member["revision"] for member in record["members"]] == ["develop", "global"]
    assert [path.parent.name for path in attempt.glob("members/*/member.json")] == ["develop", "global"]
    assert all(
        json.loads(path.read_text())["identity"]["attempt_id"] == attempt.name
        for path in attempt.glob("members/*/member.json")
    )
    assert record["identity"]["observation_key"] == observation.observation_key


def test_pair_rejection_retains_child_failure_and_shared_telemetry(tmp_path):
    """A failed child cannot erase its stderr or the telemetry shared by the pair."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_pair_failure")
    context = _coordinate_context(benchmark, tmp_path)
    context = benchmark._CoordinateContext(**{**context.__dict__, "device": "cuda:0"})
    sample = benchmark.TelemetrySample(0.0, 40.0, 0.0, 1800.0, 9000.0, "", ())
    runner = _FakeChildRunner(benchmark, fail_calls=(1,))
    coordinator = benchmark.Coordinator(
        tmp_path / "run",
        runner=runner,
        telemetry_sampler=lambda _: sample,
        worktree_probe=_clean_probe(benchmark, context),
        sleep=lambda _: None,
    )
    observation = next(
        item for item in benchmark.runtime_coordinate_schedule(2) if item.kind == "pair" and item.pair_id == "01"
    )
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert record["status"] == "rejected"
    assert len(record["telemetry"]["samples"]["pre"]) == 20
    assert len(record["telemetry"]["samples"]["post"]) == 20
    assert record["members"][0]["process"]["returncode"] == 17
    assert record["members"][0]["process"]["stderr"] == "child exploded"
    assert record["members"][1]["process"]["returncode"] == 0


def test_pair_persists_runner_exception_as_rejected_process_evidence(tmp_path):
    """A subprocess-launch exception cannot strand an unpublished attempt directory."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_runner_exception")
    context = _coordinate_context(benchmark, tmp_path)

    def runner(*_args, **_kwargs):
        raise OSError("cannot spawn wrapper")

    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=runner, worktree_probe=_clean_probe(benchmark, context), sleep=lambda _: None
    )
    observation = next(
        item for item in benchmark.runtime_coordinate_schedule(2) if item.kind == "pair" and item.pair_id == "01"
    )
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert record["status"] == "rejected"
    assert record["members"][0]["process"]["returncode"] == -1
    assert "cannot spawn wrapper" in record["members"][0]["process"]["stderr"]


def test_pair_rejects_member_payload_with_wrong_inner_revision_identity(tmp_path):
    """A correctly named child file cannot smuggle a different member identity."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_inner_identity")
    context = _coordinate_context(benchmark, tmp_path)
    base_runner = _FakeChildRunner(benchmark)

    def corrupt_runner(command, *, cwd, env):
        result = base_runner(command, cwd=cwd, env=env)
        output = Path(command[command.index("--output_path") + 1]) / "member.json"
        payload = json.loads(output.read_text())
        payload["member"]["revision"] = "current"
        output.write_text(json.dumps(payload), encoding="utf-8")
        return result

    coordinator = benchmark.Coordinator(
        tmp_path / "run",
        runner=corrupt_runner,
        worktree_probe=_clean_probe(benchmark, context),
        sleep=lambda _: None,
    )
    observation = next(
        item for item in benchmark.runtime_coordinate_schedule(2) if item.kind == "pair" and item.pair_id == "01"
    )
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert record["status"] == "rejected"
    assert "member revision identity mismatch" in record["process"]["rejection_reasons"][0]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("eager_graph", "effective execution mismatch"),
        ("unsupported", "supported capability"),
        ("missing_timing", "valid timing"),
        ("unproven_graph", "live graph capture"),
    ],
)
def test_pair_rejects_invalid_supported_or_graph_member_contract(tmp_path, mutation, expected_reason):
    """A child envelope cannot turn eager, unsupported, or untimed work into an accepted pair."""
    benchmark = _load(_BENCHMARK, f"actuator_benchmark_member_contract_{mutation}")
    context = _coordinate_context(benchmark, tmp_path)
    base_runner = _FakeChildRunner(benchmark)

    def corrupt_runner(command, *, cwd, env):
        result = base_runner(command, cwd=cwd, env=env)
        output = Path(command[command.index("--output_path") + 1]) / "member.json"
        payload = json.loads(output.read_text())
        if payload["member"]["requested_execution"] != "graph":
            return result
        if mutation == "eager_graph":
            payload["member"]["effective_execution"] = "cached_eager"
        elif mutation == "unsupported":
            payload["member"]["capability"] = {"supported": False, "reason": "capture unavailable"}
            payload["member"]["effective_execution"] = None
            payload["member"]["timing"] = None
        elif mutation == "missing_timing":
            payload["member"]["timing"] = None
        else:
            payload["member"]["execution"]["graph_capture_live"] = False
        output.write_text(json.dumps(payload), encoding="utf-8")
        return result

    coordinator = benchmark.Coordinator(
        tmp_path / "run",
        runner=corrupt_runner,
        worktree_probe=_clean_probe(benchmark, context),
        sleep=lambda _: None,
    )
    observation = next(
        item
        for item in benchmark.runtime_coordinate_schedule(2)
        if item.kind == "pair" and item.mode_pair == "develop-cached_eager__global-graph" and item.pair_id == "01"
    )
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert record["status"] == "rejected"
    assert expected_reason in " ".join(record["process"]["rejection_reasons"])


def test_measured_singleton_rejects_missing_timing(tmp_path):
    """A measured global-only singleton cannot enter selection without latency evidence."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_singleton_timing_contract")
    context = _coordinate_context(benchmark, tmp_path)
    base_runner = _FakeChildRunner(benchmark)

    def untimed_runner(command, *, cwd, env):
        result = base_runner(command, cwd=cwd, env=env)
        output = Path(command[command.index("--output_path") + 1]) / "member.json"
        payload = json.loads(output.read_text())
        payload["member"]["timing"] = None
        output.write_text(json.dumps(payload), encoding="utf-8")
        return result

    coordinator = benchmark.Coordinator(
        tmp_path / "run",
        runner=untimed_runner,
        worktree_probe=_clean_probe(benchmark, context),
        sleep=lambda _: None,
    )
    observation = next(item for item in benchmark.build_coordinate_schedule(2, 2) if item.kind == "singleton")
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert record["status"] == "rejected"
    assert "valid timing" in " ".join(record["process"]["rejection_reasons"])


def test_pair_revalidates_worktree_sha_after_each_member(tmp_path):
    """A worktree changed during a long batch cannot enter one accepted pair."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_member_revalidation")
    context = _coordinate_context(benchmark, tmp_path)
    runner = _FakeChildRunner(benchmark)
    probes = 0

    def probe(path):
        nonlocal probes
        probes += 1
        revision = path.name
        sha = context.revision_shas[revision]
        return benchmark.WorktreeState(sha, probes == 2)

    coordinator = benchmark.Coordinator(tmp_path / "run", runner=runner, worktree_probe=probe, sleep=lambda _: None)
    observation = next(
        item for item in benchmark.runtime_coordinate_schedule(2) if item.kind == "pair" and item.pair_id == "01"
    )
    attempt = coordinator.run_observation(observation, context)
    record = json.loads((attempt / "attempt.json").read_text())
    assert record["status"] == "rejected"
    assert "changed after child execution" in record["process"]["rejection_reasons"][0]


def test_rejected_atomic_pair_retry_uses_next_attempt_and_selects_only_success(tmp_path):
    """Retry allocates new evidence and leaves the first rejected pair immutable."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_atomic_retry")
    context = _coordinate_context(benchmark, tmp_path)
    runner = _FakeChildRunner(benchmark, fail_calls=(1,))
    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=runner, worktree_probe=_clean_probe(benchmark, context), sleep=lambda _: None
    )
    observation = next(
        item for item in benchmark.runtime_coordinate_schedule(2) if item.kind == "pair" and item.pair_id == "01"
    )
    selected = coordinator.run_until_selected(observation, context)
    first = selected.parent / "attempt-01" / "attempt.json"
    assert selected.name == "attempt-02"
    assert json.loads(first.read_text())["status"] == "rejected"
    manifest = json.loads((tmp_path / "run" / "accepted-attempts.json").read_text())
    assert manifest["attempts"] == [str(selected.relative_to(tmp_path / "run"))]


def test_child_wrapper_and_cache_environment_are_revision_exact(tmp_path):
    """A child must run the immutable harness through its own pinned worktree wrapper."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_child_wrapper")
    context = _coordinate_context(benchmark, tmp_path)
    runner = _FakeChildRunner(benchmark)
    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=runner, worktree_probe=_clean_probe(benchmark, context), sleep=lambda _: None
    )
    observation = next(
        item for item in benchmark.runtime_coordinate_schedule(2) if item.kind == "pair" and item.pair_id == "01"
    )
    coordinator.run_observation(observation, context)
    for command, cwd, env in runner.calls:
        revision = command[command.index("--revision") + 1]
        sha = context.revision_shas[revision]
        assert command[:3] == [str((context.worktrees[revision] / "isaaclab.sh").resolve()), "-p", str(context.harness)]
        assert cwd == context.worktrees[revision].resolve()
        assert env["WARP_CACHE_PATH"] == str((tmp_path / "cache" / sha).resolve())


def test_coordinate_rejects_existing_batch_wrong_sha_and_dirty_worktree(tmp_path):
    """Preflight must reject contaminated identities before launching a child."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_coordinate_preflight")
    args = benchmark.parse_args(_coordinate_argv(tmp_path))
    for revision in ("develop", "current", "global"):
        path = tmp_path / revision
        path.mkdir()
        (path / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (path / "uv.lock").write_text(f"lock-{revision}\n", encoding="utf-8")
    states = {
        (tmp_path / "develop").resolve(): benchmark.WorktreeState("d" * 40, False),
        (tmp_path / "current").resolve(): benchmark.WorktreeState("c" * 40, False),
        (tmp_path / "global").resolve(): benchmark.WorktreeState("a" * 40, False),
    }
    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=lambda *_args, **_kwargs: pytest.fail("child launched"), worktree_probe=states.get
    )
    (tmp_path / "run" / "batches" / "runtime-01").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="batch"):
        coordinator.coordinate(args)
    (tmp_path / "run" / "batches" / "runtime-01").rmdir()
    states[(tmp_path / "develop").resolve()] = benchmark.WorktreeState("0" * 40, False)
    with pytest.raises(ValueError, match="develop.*SHA"):
        coordinator.coordinate(args)
    states[(tmp_path / "develop").resolve()] = benchmark.WorktreeState("d" * 40, True)
    with pytest.raises(ValueError, match="develop.*dirty"):
        coordinator.coordinate(args)


def test_selection_manifest_is_atomic_with_append_only_history(tmp_path):
    """A mutable selection update must preserve every prior manifest snapshot."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_selection_history")
    context = _coordinate_context(benchmark, tmp_path)
    coordinator = benchmark.Coordinator(tmp_path / "run", sleep=lambda _: None)
    attempts = []
    for index in range(2):
        attempt = tmp_path / "run" / "observations" / f"row-{index}" / "attempt-01"
        attempt.mkdir(parents=True)
        record = _record(benchmark)
        record["identity"]["candidate_sha"] = context.candidate_sha
        record["identity"]["harness_sha256"] = context.harness_sha256
        record["identity"]["observation_key"] = f"row-{index}"
        record["identity"]["revision_shas"] = context.revision_shas
        (attempt / "attempt.json").write_text(json.dumps(record), encoding="utf-8")
        coordinator.select_attempt(attempt, context)
        attempts.append(str(attempt.relative_to(tmp_path / "run")))
    history = sorted((tmp_path / "run" / "selection-history").glob("accepted-attempts-*.json"))
    assert len(history) == 2
    assert json.loads(history[0].read_text())["attempts"] == attempts[:1]
    assert json.loads(history[1].read_text())["attempts"] == attempts
    assert json.loads((tmp_path / "run" / "accepted-attempts.json").read_text())["attempts"] == attempts


def test_selection_rejects_attempt_or_manifest_revision_sha_mixture(tmp_path):
    """Candidate and harness equality cannot hide a different historical revision map."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_selection_revision_map")
    context = _coordinate_context(benchmark, tmp_path)
    coordinator = benchmark.Coordinator(tmp_path / "run", sleep=lambda _: None)

    def write_attempt(name, observation_key, revision_shas):
        attempt = tmp_path / "run" / "observations" / name / "attempt-01"
        attempt.mkdir(parents=True)
        record = _record(benchmark)
        record["identity"].update(
            candidate_sha=context.candidate_sha,
            harness_sha256=context.harness_sha256,
            observation_key=observation_key,
            revision_shas=revision_shas,
        )
        (attempt / "attempt.json").write_text(json.dumps(record), encoding="utf-8")
        return attempt

    wrong_attempt = write_attempt("wrong-attempt", "wrong-attempt", {**context.revision_shas, "develop": "e" * 40})
    with pytest.raises(ValueError, match="attempt revision SHA"):
        coordinator.select_attempt(wrong_attempt, context)

    first = write_attempt("first", "first", context.revision_shas)
    coordinator.select_attempt(first, context)
    mixed_context = benchmark._CoordinateContext(
        **{**context.__dict__, "revision_shas": {**context.revision_shas, "current": "e" * 40}}
    )
    second = write_attempt("second", "second", mixed_context.revision_shas)
    with pytest.raises(ValueError, match="manifest revision SHA"):
        coordinator.select_attempt(second, mixed_context)


def test_fake_runner_coordinates_complete_runtime_batch(tmp_path, monkeypatch):
    """The import-safe parent produces a complete selected runtime batch through process seams."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_coordinate_e2e")
    args = benchmark.parse_args(_coordinate_argv(tmp_path, pair_repetitions=2))
    for revision in ("develop", "current", "global"):
        path = tmp_path / revision
        path.mkdir()
        (path / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (path / "uv.lock").write_text(f"lock-{revision}\n", encoding="utf-8")
    states = {
        (tmp_path / "develop").resolve(): benchmark.WorktreeState("d" * 40, False),
        (tmp_path / "current").resolve(): benchmark.WorktreeState("c" * 40, False),
        (tmp_path / "global").resolve(): benchmark.WorktreeState("a" * 40, False),
    }
    runner = _FakeChildRunner(benchmark)
    initial_metadata = {
        "python": "3.12.13",
        "platform": "test-platform",
        "gpu": [{"index": 0, "name": "Test GPU", "driver_version": "999.1"}],
        "versions": {"torch": "2.test"},
        "cuda": {
            "driver_version": "999.1",
            "runtime_version": "12.9",
            "torch_version": "2.test",
            "torch_cuda_version": "12.8",
        },
    }
    monkeypatch.setattr(benchmark, "_collect_initial_metadata", lambda: initial_metadata)
    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=runner, worktree_probe=states.get, sleep=lambda _: None
    )
    coordinator.coordinate(args)
    manifest = json.loads((tmp_path / "run" / "accepted-attempts.json").read_text())
    # 9 actuator/group rows x (four two-pair comparisons + two unsupported graph singletons).
    assert len(manifest["attempts"]) == 90
    assert len(runner.calls) == 147
    prewarm = runner.calls[:3]
    assert [call[0][call[0].index("--revision") + 1] for call in prewarm] == ["develop", "current", "global"]
    assert all(call[0][call[0].index("--phase") + 1] == "compile_prewarm" for call in prewarm)
    assert all("prewarm" not in selected for selected in manifest["attempts"])
    assert all((tmp_path / "run" / path / "attempt.json").is_file() for path in manifest["attempts"])
    first = json.loads((tmp_path / "run" / manifest["attempts"][0] / "attempt.json").read_text())
    assert first["command"][0].endswith("benchmark_actuator_collection.py")
    assert first["metadata"]["initial"]["python"]
    batch = json.loads((tmp_path / "run" / "batches" / "runtime-01" / "manifest.json").read_text())
    assert batch["command"] == first["command"]
    assert batch["initial_metadata"] == first["metadata"]["initial"]
    assert batch["worktree_states"] == {
        "develop": {"head_sha": "d" * 40, "dirty": False},
        "current": {"head_sha": "c" * 40, "dirty": False},
        "global": {"head_sha": "a" * 40, "dirty": False},
    }
    expected_locks = {
        revision: hashlib.sha256(f"lock-{revision}\n".encode()).hexdigest()
        for revision in ("develop", "current", "global")
    }
    assert batch["lockfile_sha256"] == expected_locks
    assert first["metadata"]["lockfile_sha256"] == expected_locks
    assert len(batch["benchmark_config_sha256"]) == 64
    assert first["metadata"]["benchmark_config_sha256"] == batch["benchmark_config_sha256"]
    assert batch["initial_metadata"]["cuda"]["driver_version"] is not None
    assert "runtime_version" in batch["initial_metadata"]["cuda"]
    assert "torch_cuda_version" in batch["initial_metadata"]["cuda"]
    cache_environment = first["cache"]["environment"]
    for revision, value in cache_environment.items():
        assert value == {
            "name": "WARP_CACHE_PATH",
            "value": str((tmp_path / "cache" / states[(tmp_path / revision).resolve()].head_sha).resolve()),
        }
    paired = next(
        json.loads((tmp_path / "run" / selected / "attempt.json").read_text())
        for selected in manifest["attempts"]
        if json.loads((tmp_path / "run" / selected / "attempt.json").read_text())["kind"] == "pair"
    )
    paired_member = paired["members"][0]
    assert paired_member["process"]["environment"] == {
        "WARP_CACHE_PATH": paired["cache"]["environment"][paired_member["revision"]]["value"]
    }


def test_parent_rejects_compile_prewarm_member_with_timing(tmp_path):
    """A cache-prewarm child containing latency samples cannot become accepted batch evidence."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_timed_prewarm_parent")
    context = _coordinate_context(benchmark, tmp_path)
    base_runner = _FakeChildRunner(benchmark)

    def timed_runner(command, *, cwd, env):
        result = base_runner(command, cwd=cwd, env=env)
        output = Path(command[command.index("--output_path") + 1]) / "member.json"
        payload = json.loads(output.read_text())
        payload["member"]["timing"] = {"samples_ms": [1.0]}
        output.write_text(json.dumps(payload), encoding="utf-8")
        return result

    coordinator = benchmark.Coordinator(
        tmp_path / "run", runner=timed_runner, worktree_probe=_clean_probe(benchmark, context), sleep=lambda _: None
    )
    batch = tmp_path / "run" / "batches" / "runtime-01"
    batch.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="prewarm.*timing"):
        coordinator._prewarm_revisions(context, batch)
    evidence = json.loads((batch / "prewarm" / "prewarm-develop.json").read_text())
    assert evidence["status"] == "rejected"
    assert evidence["member"]["timing"] == {"samples_ms": [1.0]}


def test_final_child_closes_adapter_and_persists_supported_failure(tmp_path, monkeypatch):
    """An adapter exception still produces exact member evidence and closes resources."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_final_child_failure")
    row = benchmark.expand_build_matrix("B1")[0]

    class Adapter:
        closed = False

        def build_workload(self, _workload):
            raise RuntimeError("construction failed")

        def close(self):
            self.closed = True

    adapter = Adapter()
    monkeypatch.setattr(benchmark, "select_adapter", lambda *_args: adapter)
    args = SimpleNamespace(
        mode="build",
        revision="global",
        revision_sha="a" * 40,
        candidate_sha="a" * 40,
        observation_key="build|B1|pair-01",
        attempt_id="attempt-03",
        phase="cold",
        child_row=json.dumps(benchmark._row_payload(row)),
        harness_sha256="f" * 64,
        batch_id="build-01",
        device="cpu",
        warmup_iterations=1,
        num_iterations=1,
        output_path=tmp_path / "member",
    )
    assert benchmark._run_final_child(args) == 1
    payload = json.loads((args.output_path / "member.json").read_text())
    assert adapter.closed is True
    assert payload["identity"]["attempt_id"] == "attempt-03"
    assert payload["revision_sha"] == "a" * 40
    assert payload["status"] == "rejected"
    assert payload["reason"] == "build_workload: RuntimeError: construction failed"


def test_final_child_compile_prewarm_uses_explicit_untimed_contract(tmp_path, monkeypatch):
    """Compile prewarm calls its dedicated adapter hook and emits no timing observation."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_final_child_prewarm")
    row = benchmark.expand_build_matrix("B1")[0]

    class Adapter(benchmark._MemoryAdapter):
        prewarm_calls = 0

        def build_workload(self, _workload):
            raise AssertionError("measured build path used for compile prewarm")

        def compile_prewarm(self, workload):
            self.prewarm_calls += 1
            self.workload = workload

    adapter = Adapter("global", "cpu")
    monkeypatch.setattr(benchmark, "select_adapter", lambda *_args: adapter)
    args = SimpleNamespace(
        mode="build",
        revision="global",
        revision_sha="a" * 40,
        candidate_sha="a" * 40,
        observation_key="prewarm|global",
        attempt_id="prewarm",
        phase="compile_prewarm",
        child_row=json.dumps(benchmark._row_payload(row)),
        harness_sha256="f" * 64,
        batch_id="build-01",
        device="cpu",
        warmup_iterations=1,
        num_iterations=1,
        output_path=tmp_path / "member",
    )
    assert benchmark._run_final_child(args) == 0
    payload = json.loads((args.output_path / "member.json").read_text())
    assert adapter.prewarm_calls == 1
    assert payload["status"] == "accepted"
    assert payload["member"]["requested_execution"] == "compile_prewarm"
    assert payload["member"]["effective_execution"] == "compile_prewarm"
    assert payload["member"]["timing"] is None


def test_final_run_cli_writes_real_member_contract_with_exact_identity(tmp_path, monkeypatch):
    """The actual child CLI, not only the fake runner, writes the parent-consumed schema."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_final_child_contract")
    row = benchmark.expand_build_matrix("B1")[0]

    class Adapter(benchmark._MemoryAdapter):
        pass

    monkeypatch.setattr(benchmark, "select_adapter", lambda revision, device: Adapter(revision, device))
    output = tmp_path / "members" / "global"
    argv = [
        "--mode",
        "build",
        "--revision",
        "global",
        "--revision_sha",
        "a" * 40,
        "--candidate_sha",
        "a" * 40,
        "--observation_key",
        "build|B1|global",
        "--attempt_id",
        "attempt-07",
        "--phase",
        "cold",
        "--child_row",
        json.dumps(benchmark._row_payload(row)),
        "--harness_sha256",
        "f" * 64,
        "--batch_id",
        "build-01",
        "--final_run",
        "--device",
        "cpu",
        "--output_path",
        str(output),
    ]
    assert benchmark.main(argv) == 0
    payload = json.loads((output / "member.json").read_text())
    assert payload["schema"] == "actuator_collection_member/v1"
    assert payload["identity"] == {
        "batch_id": "build-01",
        "observation_key": "build|B1|global",
        "attempt_id": "attempt-07",
        "candidate_sha": "a" * 40,
        "harness_sha256": "f" * 64,
    }
    assert payload["member"]["revision"] == "global"
    assert payload["member"]["resolved_row"] == benchmark._row_payload(row)
    assert payload["metadata"]["python"]


def test_gpu_telemetry_window_has_twenty_samples_and_nineteen_cadence_sleeps(tmp_path):
    """Changing the cadence or sample count invalidates shared pair evidence."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_telemetry_window")
    sleeps = []
    sample = benchmark.TelemetrySample(0.0, 40.0, 0.0, 1800.0, 9000.0, "", ())
    coordinator = benchmark.Coordinator(
        tmp_path, telemetry_sampler=lambda index: sample, sleep=lambda seconds: sleeps.append(seconds)
    )
    assert coordinator._sample_window("cuda:2") == [sample] * 20
    assert sleeps == [0.25] * 19


def test_gpu_telemetry_falls_back_to_nvidia_smi_with_device_index(monkeypatch):
    """Unavailable NVML must retain complete metrics through the defensive CLI path."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_telemetry_fallback")
    monkeypatch.setitem(sys.modules, "pynvml", None)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if "--query-compute-apps=pid" in command:
            return SimpleNamespace(returncode=0, stdout="123\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="41, 2, 1800, 9000, 0x0000000000000000\n", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    sample = benchmark._TelemetrySampler().sample(2)
    assert all("--id=2" in command for command in calls)
    assert sample.temperature_c == 41.0
    assert sample.utilization_pct == 2.0
    assert sample.throttle_reasons == ""
    assert sample.compute_pids == (123,)


def test_initial_metadata_records_structured_cuda_driver_runtime_and_torch_identity(monkeypatch):
    """Final provenance distinguishes the NVIDIA driver, CUDA runtime, and Torch CUDA build."""
    benchmark = _load(_BENCHMARK, "actuator_benchmark_cuda_provenance")

    def run(command, **_kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="0, Test GPU, 555.42\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "torch_version": "2.test",
                    "torch_cuda_version": "12.8",
                    "runtime_version": "12.9",
                    "driver_version": "13.0",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    metadata = benchmark._collect_initial_metadata()
    assert metadata["gpu"] == [{"index": 0, "name": "Test GPU", "driver_version": "555.42"}]
    assert metadata["cuda"] == {
        "driver_version": "555.42",
        "runtime_version": "12.9",
        "warp_driver_version": "13.0",
        "torch_version": "2.test",
        "torch_cuda_version": "12.8",
        "probe_error": None,
    }


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
    sample = benchmark.TelemetrySample(0.0, 40.0, 0.0, 1000.0, 1000.0, "", ())
    assert benchmark.validate_pair_telemetry([sample] * 19, [sample] * 20, "cuda:0") == [
        "required telemetry unavailable"
    ]
    assert benchmark.validate_pair_telemetry([sample] * 20, [sample] * 20, "cuda:0") == []
