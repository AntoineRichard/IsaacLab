# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the persistent benchmark host-idle gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmark_comparison.runner import (
    HostIdleGate,
    IdleGateTimeout,
    IdleInventory,
    IdleSample,
    IdleThresholds,
    SystemIdleMonitor,
)


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _Monitor:
    def __init__(self, rounds: list[tuple[IdleInventory, list[IdleSample]]]):
        self.rounds = rounds
        self.round_index = -1
        self.sample_index = 0
        self.inventory_count = 0

    def inventory(self) -> IdleInventory:
        if self.inventory_count % 2 == 0:
            self.round_index += 1
            self.sample_index = 0
        self.inventory_count += 1
        return self.rounds[self.round_index][0]

    def sample(self) -> IdleSample:
        sample = self.rounds[self.round_index][1][self.sample_index]
        self.sample_index += 1
        return sample


def _samples(*, util: int = 2, memory: int = 300, load: float = 1.0) -> list[IdleSample]:
    return [
        IdleSample(index=index, gpu_utilization_pct=util, gpu_memory_mib=memory, load_1m=load) for index in range(60)
    ]


def _inventory(*, compute=(), containers=(), children=()) -> IdleInventory:
    return IdleInventory(
        nvidia_compute_processes=tuple(compute),
        gpu_container_ids=tuple(containers),
        prior_child_pids=tuple(children),
    )


def _gate(tmp_path: Path, rounds, clock: _Clock, timeout: float = 900.0) -> HostIdleGate:
    return HostIdleGate(
        monitor=_Monitor(rounds),
        clock=clock,
        evidence_root=tmp_path,
        idle_memory_baseline_mib=256,
        logical_cpu_count=8,
        thresholds=IdleThresholds(sample_count=60, sample_interval_s=1, retry_interval_s=300, timeout_s=timeout),
    )


def test_idle_gate_accepts_sixty_consecutive_samples_and_persists_raw_evidence(tmp_path: Path):
    clock = _Clock()
    gate = _gate(tmp_path, [(_inventory(), _samples())], clock)

    evidence = gate.wait("attempt-a")

    document = json.loads(evidence.read_text(encoding="utf-8"))
    assert document["decision"] == "accepted"
    assert len(document["samples"]) == 60
    assert document["idle_memory_baseline_mib"] == 256
    assert document["thresholds"]["gpu_utilization_max_pct"] == 5
    assert document["thresholds"]["gpu_memory_headroom_mib"] == 1024
    assert document["thresholds"]["load_1m_max"] == 2.0
    assert clock.sleeps == [1] * 59


@pytest.mark.parametrize(
    ("inventory", "samples", "reason"),
    [
        (_inventory(compute=(123,)), _samples(), "nvidia_compute_process"),
        (_inventory(containers=("container-a",)), _samples(), "gpu_container"),
        (_inventory(children=(456,)), _samples(), "prior_child"),
        (_inventory(), _samples(util=6), "gpu_utilization"),
        (_inventory(), _samples(memory=1281), "gpu_memory"),
        (_inventory(), _samples(load=2.1), "host_load"),
    ],
)
def test_idle_gate_rejects_busy_gpu_or_host_and_times_out(
    tmp_path: Path, inventory: IdleInventory, samples: list[IdleSample], reason: str
):
    clock = _Clock()
    gate = _gate(tmp_path, [(inventory, samples)], clock, timeout=60)

    with pytest.raises(IdleGateTimeout):
        gate.wait("attempt-a")

    document = json.loads(next(tmp_path.glob("attempt-a-idle-*.json")).read_text(encoding="utf-8"))
    assert document["decision"] == "rejected"
    assert reason in document["reasons"]
    assert len(document["samples"]) == 60


def test_idle_gate_waits_five_minutes_then_retries(tmp_path: Path):
    clock = _Clock()
    rounds = [
        (_inventory(), _samples(util=20)),
        (_inventory(), _samples(util=1)),
    ]
    gate = _gate(tmp_path, rounds, clock, timeout=1000)

    evidence = gate.wait("attempt-a")

    assert evidence.name == "attempt-a-idle-0002.json"
    assert 300 in clock.sleeps
    first = json.loads((tmp_path / "attempt-a-idle-0001.json").read_text(encoding="utf-8"))
    second = json.loads(evidence.read_text(encoding="utf-8"))
    assert first["decision"] == "rejected"
    assert second["decision"] == "accepted"


def test_idle_gate_reinventories_after_final_sample_before_accepting(tmp_path: Path):
    class LateProcessMonitor:
        def __init__(self):
            self.inventory_count = 0
            self.sample_count = 0

        def inventory(self):
            self.inventory_count += 1
            return _inventory(compute=(999,)) if self.inventory_count == 2 else _inventory()

        def sample(self):
            sample = _samples()[self.sample_count]
            self.sample_count += 1
            return sample

    clock = _Clock()
    gate = HostIdleGate(
        monitor=LateProcessMonitor(),
        clock=clock,
        evidence_root=tmp_path,
        idle_memory_baseline_mib=256,
        logical_cpu_count=8,
        thresholds=IdleThresholds(sample_count=60, sample_interval_s=1, retry_interval_s=300, timeout_s=60),
    )

    with pytest.raises(IdleGateTimeout):
        gate.wait("attempt-a")

    document = json.loads((tmp_path / "attempt-a-idle-0001.json").read_text(encoding="utf-8"))
    assert document["inventory_after_samples"]["nvidia_compute_processes"] == [999]
    assert "nvidia_compute_process" in document["reasons"]


def test_idle_evidence_ids_are_monotonic_across_resume_and_never_overwrite(tmp_path: Path):
    clock = _Clock()
    gate = _gate(
        tmp_path,
        [(_inventory(), _samples()), (_inventory(), _samples())],
        clock,
    )

    first = gate.wait("attempt-a")
    first_contents = first.read_bytes()
    second = gate.wait("attempt-a")

    assert first.name == "attempt-a-idle-0001.json"
    assert second.name == "attempt-a-idle-0002.json"
    assert first.read_bytes() == first_contents


def test_idle_evidence_scan_accepts_arbitrary_width_ids(tmp_path: Path):
    (tmp_path / "attempt-a-idle-10000.json").write_text("preserved\n", encoding="utf-8")
    gate = _gate(tmp_path, [(_inventory(), _samples())], _Clock())

    evidence = gate.wait("attempt-a")

    assert evidence.name == "attempt-a-idle-10001.json"
    assert (tmp_path / "attempt-a-idle-10000.json").read_text(encoding="utf-8") == "preserved\n"


def test_system_idle_monitor_reports_all_owned_process_group_descendants():
    class OwnedGroups:
        def alive_pids(self):
            return (401, 402, 499)

    monitor = SystemIdleMonitor(owned_process_groups=OwnedGroups())
    monitor._nvidia_compute_processes = lambda: ()
    monitor._gpu_container_ids = lambda: ()

    inventory = monitor.inventory()

    assert inventory.prior_child_pids == (401, 402, 499)
