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
from .import_results import import_completed_attempts, preflight_import_paths
from .manifest import write_manifest
from .matrix import expand_canary_matrix, expand_final_matrix, load_matrix
from .models import RunSet
from .runner import (
    BenchmarkRunner,
    ControllerLock,
    HostIdleGate,
    IdleThresholds,
    OwnedProcessGroups,
    SystemClock,
    SystemIdleMonitor,
)


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
    parser.add_argument("--import_from_artifact_root", type=Path)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--idle_timeout_s", type=float, default=3600)
    args = parser.parse_args(argv)
    if args.prepare_only and args.import_from_artifact_root is None:
        parser.error("--prepare_only requires --import_from_artifact_root")

    run_set = RunSet(args.run_set)
    if args.import_from_artifact_root is not None:
        with preflight_import_paths(args.import_from_artifact_root, args.artifact_root, run_set) as import_paths:
            with ControllerLock(import_paths.destination_root):
                config = _executor_config(args, import_paths.destination_root.resolve())
                return _run_locked(
                    config,
                    args,
                    run_set,
                    import_source_root=import_paths.source_root,
                    import_destination_root=import_paths.destination_root,
                    manifest_path=import_paths.destination_run_set / "manifest.json",
                    preflight_artifact_root=import_paths.destination_root,
                )

    config = _executor_config(args, args.artifact_root.resolve())
    with ControllerLock(config.artifact_root):
        return _run_locked(config, args, run_set)


def _executor_config(args: argparse.Namespace, artifact_root: Path) -> ExecutorConfig:
    """Build executor configuration around the selected artifact-root view."""
    return ExecutorConfig(
        lab2_root=args.lab2_root.resolve(),
        lab3_root=args.lab3_root.resolve(),
        artifact_root=artifact_root,
        lab2_sha=args.lab2_sha,
        lab3_sha=args.lab3_sha,
        lab2_image=args.lab2_image,
        lab2_image_id=args.lab2_image_id,
    )


def _run_locked(
    config: ExecutorConfig,
    args: argparse.Namespace,
    run_set: RunSet,
    *,
    import_source_root: Path | None = None,
    import_destination_root: Path | None = None,
    manifest_path: Path | None = None,
    preflight_artifact_root: Path | None = None,
) -> int:
    """Run preflight and the selected matrix while holding the controller lock."""
    preflight = (
        run_preflight(config)
        if preflight_artifact_root is None
        else run_preflight(config, artifact_root_for_writes=preflight_artifact_root)
    )
    expansion = expand_canary_matrix(load_matrix()) if run_set is RunSet.CANARY else expand_final_matrix(load_matrix())
    write_manifest(
        manifest_path or config.artifact_root / run_set.value / "manifest.json",
        preflight.manifest(run_set, args.phase, expansion),
    )
    if args.import_from_artifact_root is not None:
        if import_source_root is None or import_destination_root is None:
            raise RuntimeError("validated import paths are missing")
        import_completed_attempts(import_source_root, import_destination_root, run_set)
    if args.prepare_only:
        return 0
    if import_destination_root is not None:
        _validate_measured_artifact_root(config.artifact_root, import_destination_root)
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


def _validate_measured_artifact_root(canonical_root: Path, retained_root: Path) -> None:
    """Require the executor-facing root to remain the retained destination inode."""
    try:
        canonical_descriptor = os.open(
            canonical_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ValueError(f"measured artifact root is unavailable, replaced, or symlinked: {canonical_root}") from error
    try:
        try:
            retained_descriptor = os.open(retained_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError as error:
            raise ValueError(f"retained measured artifact root is unavailable: {retained_root}") from error
        try:
            canonical_metadata = os.fstat(canonical_descriptor)
            retained_metadata = os.fstat(retained_descriptor)
        finally:
            os.close(retained_descriptor)
    finally:
        os.close(canonical_descriptor)
    if (canonical_metadata.st_dev, canonical_metadata.st_ino) != (
        retained_metadata.st_dev,
        retained_metadata.st_ino,
    ):
        raise ValueError("measured artifact root changed after import")


if __name__ == "__main__":
    raise SystemExit(main())
