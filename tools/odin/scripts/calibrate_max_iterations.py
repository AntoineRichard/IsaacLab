# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate per-task max_iterations against a reference GPU.

For each kept task in physx_envs.yaml: dispatch a short calibration run
(``max_iterations=200``) on the configured fleet, read the resulting bundle
to extract per-iter wall-time and one-time startup, then compute the
``max_iterations`` that fits inside ``0.75 × per_job_timeout_s``.

Writes the recommendations back into the YAML alongside a ``calibrated_at``
timestamp and reference-GPU name.

Usage::

    PYTHONPATH=. python3 tools/odin/scripts/calibrate_max_iterations.py \\
        --fleet fleet.yaml \\
        --yaml tools/odin/config/physx_envs.yaml \\
        --reference-gpu L40 \\
        --per-job-timeout 43200
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["recommend_max_iterations", "_parse_calibration_metrics"]


def recommend_max_iterations(*, per_iter_s: float, startup_s: float, per_job_timeout_s: int) -> int:
    """Return the integer max_iterations that fits inside 75% of the timeout.

    Args:
        per_iter_s: Mean wall-time per training iter (from training.json).
        startup_s: One-time startup cost (from startup.json).
        per_job_timeout_s: Hard per-job timeout in seconds.

    Returns:
        Integer max_iterations such that
        ``startup_s + recommended × per_iter_s ≤ 0.75 × per_job_timeout_s``.

    Raises:
        ValueError: per_iter_s ≤ 0, or startup_s already exceeds the budget.
    """
    if per_iter_s <= 0:
        raise ValueError(f"per_iter_s must be positive, got {per_iter_s}")
    budget_s = 0.75 * per_job_timeout_s
    headroom = budget_s - startup_s
    if headroom <= 0:
        raise ValueError(f"startup_s={startup_s:.1f} exceeds budget {budget_s:.1f} (0.75 × {per_job_timeout_s}s)")
    return math.floor(headroom / per_iter_s)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, data: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _kept_rows(yaml_data: dict):
    """Yield rows from physx_envs.yaml that have ``keep: true``."""
    for _group_name, rows in (yaml_data.get("groups") or {}).items():
        for row in rows:
            if row.get("keep"):
                yield row


def _parse_calibration_metrics(bundle: Path) -> tuple[float, float]:
    """Extract (per_iter_s, startup_s) from a completed bundle.

    Args:
        bundle: Local path to a completed dispatch bundle.

    Returns:
        Tuple of ``(per_iter_s, startup_s)``.

    Raises:
        KeyError: When the bundle's training/startup JSON is missing the
            expected field paths (e.g., schema drift).
    """
    import json

    training = json.loads((bundle / "training.json").read_text())
    startup = json.loads((bundle / "startup.json").read_text())
    per_iter_s = float(training["runtime"]["iteration_time_s"]["mean"])
    startup_s = float(startup["run"]["duration_s"])
    return per_iter_s, startup_s


def _calibrate_one(
    *,
    row: dict,
    fleet,
    calibration_iters: int,
    yaml_path: Path,
    ssh,
    rsync,
    runs_root: Path,
) -> tuple[float, float] | None:
    """Run one calibration dispatch; return (per_iter_s, startup_s) or None."""
    from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch

    # TODO(scope: future PR): calibration_iters is accepted but unused here —
    # honoring it requires a runner-level max_iterations override.
    d = resolve_dispatch_dir(runs_root, resume=None)
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=yaml_path,
        newton_yaml=None,
        dispatch_dir=d,
        options=DispatchOptions(
            seeds=[42],
            per_job_timeout_s=3600,  # 1h is plenty for a 200-iter calibration
            include_filter=[row["task_id"]],
            skip_aggregate=True,
        ),
        ssh=ssh,
        rsync=rsync,
    )
    completed = [j for j in state.jobs if j.status == "completed"]
    if not completed:
        return None
    bundle = d / completed[0].bundle_dir_name
    return _parse_calibration_metrics(bundle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fleet", required=True, type=Path)
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--reference-gpu", required=True)
    parser.add_argument("--per-job-timeout", type=int, default=43200)
    parser.add_argument("--calibration-iters", type=int, default=200)
    parser.add_argument("--runs-root", type=Path, default=Path("odin_runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from tools.odin.asgard.fleet import load_fleet
    from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner

    fleet = load_fleet(args.fleet)
    ssh = ShellSSHRunner()
    rsync = ShellRsyncRunner()

    data = _load_yaml(args.yaml)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for row in _kept_rows(data):
        try:
            measurement = _calibrate_one(
                row=row,
                fleet=fleet,
                calibration_iters=args.calibration_iters,
                yaml_path=args.yaml,
                ssh=ssh,
                rsync=rsync,
                runs_root=args.runs_root,
            )
        except Exception as exc:  # noqa: BLE001 — per-task failure must not abort the sweep
            print(f"[ERROR] {row['task_id']}: {exc}")
            continue
        if measurement is None:
            print(f"[WARN] {row['task_id']}: calibration failed (no completed bundle)")
            continue
        per_iter_s, startup_s = measurement
        try:
            recommended = recommend_max_iterations(
                per_iter_s=per_iter_s,
                startup_s=startup_s,
                per_job_timeout_s=args.per_job_timeout,
            )
        except ValueError as exc:
            print(f"[WARN] {row['task_id']}: {exc} — task cannot fit, manual review needed")
            row["calibration_warning"] = str(exc)
            continue

        prior = row.get("max_iterations")
        row["max_iterations"] = recommended
        row["calibrated_at"] = now_iso
        row["calibration_reference_gpu"] = args.reference_gpu
        row["calibration_per_iter_s"] = round(per_iter_s, 4)
        row["calibration_startup_s"] = round(startup_s, 2)
        print(
            f"[OK]   {row['task_id']}: {prior} → {recommended} (per_iter={per_iter_s:.3f}s, startup={startup_s:.1f}s)"
        )

    if not args.dry_run:
        _write_yaml(args.yaml, data)
        print(f"[INFO] wrote {args.yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
