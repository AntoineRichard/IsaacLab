# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Munin — Odin's SKRL runner wrapper.

Near-verbatim companion to :mod:`tools.odin.hugin.run` with the training
script swapped to ``benchmark_skrl.py`` and the framework set to ``skrl``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from isaaclab.test.benchmark.standard_schema import ManifestPhase

from tools.odin.common.log_tail import tail_bytes
from tools.odin.common.manifest import write_manifest
from tools.odin.common.run_id import compute_run_id

# Module-level alias so tests can monkey-patch subprocess.run for Munin only.
_subprocess_run = subprocess.run

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_STARTUP_SCRIPT = os.path.join(_REPO_ROOT, "scripts/benchmarks/benchmark_startup.py")
_TRAINING_SCRIPT = os.path.join(_REPO_ROOT, "scripts/benchmarks/benchmark_skrl.py")
_ISAACLAB_SH = os.path.join(_REPO_ROOT, "isaaclab.sh")


def _run_phase(cmd: list[str], bundle_dir: str, phase_name: str, output_json: str) -> ManifestPhase:
    """Run one subprocess phase; capture exit code, duration, and log tails on failure."""
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    start = datetime.now(timezone.utc)
    completed = _subprocess_run(cmd, capture_output=True)
    end = datetime.now(timezone.utc)
    duration_s = (end - start).total_seconds()
    if completed.returncode != 0:
        status = "failed"
        with open(os.path.join(logs_dir, f"{phase_name}.stderr.log"), "wb") as f:
            f.write(tail_bytes(completed.stderr))
        with open(os.path.join(logs_dir, f"{phase_name}.stdout.log"), "wb") as f:
            f.write(tail_bytes(completed.stdout))
    else:
        status = "completed"
    return ManifestPhase(
        file=os.path.basename(output_json),
        status=status,
        duration_s=duration_s,
        exit_code=completed.returncode,
    )


def main():
    parser = argparse.ArgumentParser(description="Odin Munin — SKRL runner wrapper.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", choices=["physx", "newton"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--runs_root", type=str, default="./odin_runs")
    parser.add_argument("--ema_alpha", type=float, default=0.05)
    parser.add_argument("--no_series", action="store_true", default=False)
    parser.add_argument(
        "--skip_startup",
        action="store_true",
        default=False,
        help="Skip the dense startup-profile subprocess (training-only run).",
    )
    args = parser.parse_args()

    run_start = datetime.now(timezone.utc)
    run_id = compute_run_id("skrl", args.backend, args.task, args.seed, now=run_start)
    bundle_dir = os.path.abspath(os.path.join(args.runs_root, run_id))
    os.makedirs(bundle_dir, exist_ok=True)

    startup_phase = ManifestPhase(file="startup.json", status="completed", duration_s=0.0, exit_code=0)
    if not args.skip_startup:
        startup_phase = _run_phase(
            cmd=[
                _ISAACLAB_SH,
                "-p",
                _STARTUP_SCRIPT,
                "--task",
                args.task,
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(args.seed),
                "--headless",
                "--backend",
                args.backend,
                "--run_id",
                run_id,
                "--schema_v1_output",
                os.path.join(bundle_dir, "startup.json"),
            ],
            bundle_dir=bundle_dir,
            phase_name="startup",
            output_json=os.path.join(bundle_dir, "startup.json"),
        )

    training_data_dir = os.path.join(bundle_dir, "training_data")
    os.makedirs(training_data_dir, exist_ok=True)
    training_cmd = [
        _ISAACLAB_SH,
        "-p",
        _TRAINING_SCRIPT,
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.max_iterations),
        "--headless",
        "--backend",
        args.backend,
        "--run_id",
        run_id,
        "--schema_v1_output",
        os.path.join(bundle_dir, "training.json"),
        "--ema_alpha",
        str(args.ema_alpha),
        "--log_dir",
        training_data_dir,
    ]
    if args.no_series:
        training_cmd.append("--no_series")

    training_phase = _run_phase(
        cmd=training_cmd,
        bundle_dir=bundle_dir,
        phase_name="training",
        output_json=os.path.join(bundle_dir, "training.json"),
    )

    run_end = datetime.now(timezone.utc)
    write_manifest(
        bundle_dir=bundle_dir,
        run_id=run_id,
        framework="skrl",
        backend=args.backend,
        task=args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        run_start_dt=run_start,
        run_end_dt=run_end,
        startup_phase=startup_phase,
        training_phase=training_phase,
        repo_root=_REPO_ROOT,
    )

    if startup_phase.exit_code != 0 or training_phase.exit_code != 0:
        sys.exit(max(startup_phase.exit_code, training_phase.exit_code))


if __name__ == "__main__":
    main()
