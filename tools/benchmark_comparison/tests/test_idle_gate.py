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

    def inventory(self) -> IdleInventory:
        self.round_index += 1
        self.sample_index = 0
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
