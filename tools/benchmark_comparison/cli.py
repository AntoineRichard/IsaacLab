# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for the serialized comparison runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .executors import ExecutorConfig, Lab2DockerExecutor, Lab3UvExecutor, ProcessLauncher, run_preflight
from .manifest import write_manifest
from .matrix import expand_canary_matrix, expand_final_matrix, load_matrix
from .models import RunSet
from .runner import BenchmarkRunner, HostIdleGate, IdleThresholds, OwnedProcessGroups, SystemClock, SystemIdleMonitor


def main(argv: list[str] | None = None) -> int:
    """Run a preflighted canary or final comparison matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_set", choices=[value.value for value in RunSet], required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--lab2_root", type=Path, required=True)
    parser.add_argument("--lab3_root", type=Path, required=True)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--lab2_sha", required=True)
    parser.add_argument("--lab3_sha", required=True)
    parser.add_argument("--lab2_image", required=True)
    parser.add_argument("--lab2_image_id", required=True)
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--idle_timeout_s", type=float, default=3600)
    args = parser.parse_args(argv)

    config = ExecutorConfig(
        lab2_root=args.lab2_root.resolve(),
        lab3_root=args.lab3_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        lab2_sha=args.lab2_sha,
        lab3_sha=args.lab3_sha,
        lab2_image=args.lab2_image,
        lab2_image_id=args.lab2_image_id,
    )
    preflight = run_preflight(config)
    run_set = RunSet(args.run_set)
    expansion = expand_canary_matrix(load_matrix()) if run_set is RunSet.CANARY else expand_final_matrix(load_matrix())
    write_manifest(
        config.artifact_root / run_set.value / "manifest.json",
        preflight.manifest(run_set, args.phase, expansion),
    )
    owned_process_groups = OwnedProcessGroups()
    launcher = ProcessLauncher(owned_process_groups=owned_process_groups)
    idle_gate = HostIdleGate(
        monitor=SystemIdleMonitor(owned_process_groups=owned_process_groups),
        clock=SystemClock(),
        evidence_root=config.artifact_root / args.run_set / "idle",
        idle_memory_baseline_mib=preflight.idle_memory_baseline_mib,
        logical_cpu_count=os.cpu_count() or 1,
        thresholds=IdleThresholds(timeout_s=args.idle_timeout_s),
    )
    runner = BenchmarkRunner(
        artifact_root=config.artifact_root,
        executors={
            "lab2": Lab2DockerExecutor(
                config,
                launcher=launcher,
                provenance=preflight.provenance,
                selected_gpu_uuid=preflight.host.gpu_uuid,
            ),
            "lab3": Lab3UvExecutor(
                config,
                launcher=launcher,
                provenance=preflight.provenance,
                selected_gpu_uuid=preflight.host.gpu_uuid,
            ),
        },
        idle_gate=idle_gate,
        expected_provenance=preflight.provenance,
        expected_gpu_uuid=preflight.host.gpu_uuid,
    )
    result = runner.run(expansion, retry_failures=args.retry_failures)
    return 0 if result.failed == 0 and result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
