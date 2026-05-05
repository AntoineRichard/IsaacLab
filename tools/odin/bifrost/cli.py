# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``odin-bifrost-dispatch`` CLI entry point.

Submits a single OSMO workflow with one task per ``(env, seed)`` row from
the curated env YAMLs. Bundles return as datasets and land at
``odin_runs/<dispatch_id>/<run_id>/`` — the canonical layout.

T13 implements: arg parsing, planner, workflow rendering, dispatch.json
write, ``--dry-run``. T14 wires submit + poll. T15 adds ``--resume``.
T16 adds ``--retry-failed``. T17 adds ``--verbose`` log tail.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
    DispatchState,
    write_dispatch_state,
)
from tools.odin.bifrost.bundle import download_and_validate_bundle
from tools.odin.bifrost.client import OsmoClient
from tools.odin.bifrost.config import load_bifrost_config
from tools.odin.bifrost.poller import poll_until_terminal
from tools.odin.bifrost.workflow import (
    RenderRow,
    osmo_safe_task_name,
    render_workflow_yaml,
    stage_source_tarball,
)


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odin-bifrost-dispatch",
        description="Dispatch Odin eval jobs to OSMO as a single workflow with N parallel tasks.",
    )
    p.add_argument("--osmo-config", required=True, type=Path, help="Path to bifrost-osmo.yaml.")
    p.add_argument("--physx-yaml", required=True, type=Path, help="Curated physx env list.")
    p.add_argument("--newton-yaml", type=Path, default=None, help="Curated newton env list (optional).")
    p.add_argument("--seeds", required=True, type=_parse_seeds, help="Comma-separated seeds.")
    p.add_argument("--include", type=str, default=None, help="Glob filter on task_id.")
    p.add_argument("--pool", type=str, default=None, help="Override config.pool.")
    p.add_argument("--priority", choices=["HIGH", "NORMAL", "LOW"], default=None)
    p.add_argument("--rsync", action="store_true", help="Enable continuous rsync of source_root for dev.")
    p.add_argument("--dry-run", action="store_true", help="Render workflow YAML and exit; do not submit.")
    p.add_argument("--resume", type=str, default=None, help="<dispatch_id> | LATEST")
    p.add_argument("--retry-failed", type=str, default=None, help="Comma-separated run_ids.")
    p.add_argument("--poll-interval", type=int, default=15, help="Seconds between OSMO status polls.")
    p.add_argument("--runs-root", type=Path, default=Path("./odin_runs"))
    p.add_argument("--verbose", action="store_true")
    return p


@dataclass(frozen=True)
class _PlannedRow:
    run_id: str
    task_id: str
    framework: str  # rsl-rl | skrl
    backend: str  # physx | newton
    seed: int
    num_envs: int
    max_iterations: int


def _load_envs_yaml(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return [e for e in (data.get("envs") or []) if e.get("keep") is True]


def _matches_include(task_id: str, include_glob: str | None) -> bool:
    if not include_glob:
        return True
    return fnmatch.fnmatch(task_id, include_glob)


def _build_rows(
    *,
    physx_yaml: Path,
    newton_yaml: Path | None,
    seeds: list[int],
    include_glob: str | None,
    dispatch_id: str,
) -> list[_PlannedRow]:
    rows: list[_PlannedRow] = []
    for path, backend in [(physx_yaml, "physx"), (newton_yaml, "newton")]:
        if path is None:
            continue
        for env in _load_envs_yaml(path):
            task_id = str(env["task_id"])
            if not _matches_include(task_id, include_glob):
                continue
            framework = str(env["framework"])
            num_envs = int(env["num_envs"])
            max_iter = int(env["max_iterations"])
            for seed in seeds:
                run_id = f"{framework}_{backend}_{task_id}_{dispatch_id}_seed{seed}"
                rows.append(
                    _PlannedRow(
                        run_id=run_id,
                        task_id=task_id,
                        framework=framework,
                        backend=backend,
                        seed=seed,
                        num_envs=num_envs,
                        max_iterations=max_iter,
                    )
                )
    return rows


def _planned_to_render(row: _PlannedRow) -> RenderRow:
    return RenderRow(
        run_id=row.run_id,
        osmo_task_name=osmo_safe_task_name(row.run_id),
        framework=row.framework,
        framework_runner="hugin" if row.framework == "rsl-rl" else "munin",
        task_id=row.task_id,
        backend=row.backend,
        seed=row.seed,
        num_envs=row.num_envs,
        max_iterations=row.max_iterations,
    )


def _planned_to_job(row: _PlannedRow) -> JobEntry:
    return JobEntry(
        run_id=row.run_id,
        task_id=row.task_id,
        framework=row.framework,
        backend=row.backend,
        num_envs=row.num_envs,
        max_iterations=row.max_iterations,
        seed=row.seed,
        bundle_dir_name=row.run_id,
        status="pending",
        osmo_task_name=osmo_safe_task_name(row.run_id),
    )


def _allocate_dispatch_id(now: dt.datetime | None = None) -> str:
    return (now or dt.datetime.now(dt.UTC)).strftime("%Y%m%d-%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_bifrost_config(args.osmo_config)
    if args.pool:
        cfg = replace(cfg, pool=args.pool)
    if args.priority:
        cfg = replace(cfg, priority=args.priority)

    dispatch_id = _allocate_dispatch_id()
    dispatch_dir = args.runs_root / dispatch_id
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(
        physx_yaml=args.physx_yaml,
        newton_yaml=args.newton_yaml,
        seeds=args.seeds,
        include_glob=args.include,
        dispatch_id=dispatch_id,
    )
    if not rows:
        print("No keep:true rows matched the include filter.", file=sys.stderr)
        return 2

    tarball_path: str | None = None
    if cfg.code_delivery.mode == "files_upload":
        tarball_path_p = dispatch_dir / "odin-source.tar.gz"
        repo_root = Path.cwd()
        stage_source_tarball(repo_root / cfg.code_delivery.source_root, tarball_path_p, repo_root=repo_root)
        tarball_path = str(tarball_path_p)

    workflow_yaml = render_workflow_yaml(
        dispatch_id=dispatch_id,
        rows=[_planned_to_render(r) for r in rows],
        cfg=cfg,
        tarball_path=tarball_path,
    )
    workflow_yaml_path = dispatch_dir / "workflow.yaml"
    workflow_yaml_path.write_text(workflow_yaml)

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id=dispatch_id,
        started_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ended_at=None,
        seeds=list(args.seeds),
        commit_sha="",
        fleet=[],
        jobs=[_planned_to_job(r) for r in rows],
        dispatcher="osmo",
        osmo_workflow_id=None,
    )
    write_dispatch_state(dispatch_dir, state)

    if args.dry_run:
        print(f"[dry-run] wrote {workflow_yaml_path}")
        return 0

    client = OsmoClient(profile=cfg.osmo_profile)
    rsync_pairs: list[tuple[str, str]] = []
    if args.rsync:
        rsync_pairs.append((cfg.code_delivery.source_root, "/workspace/IsaacLab/" + cfg.code_delivery.source_root))
    workflow_id = client.submit(workflow_yaml_path, rsync_pairs=rsync_pairs)
    state.osmo_workflow_id = workflow_id
    write_dispatch_state(dispatch_dir, state)

    validator = _manifest_validator()

    def on_completed(job: JobEntry) -> None:
        dataset_name = f"{cfg.bundle_dataset_prefix}-{dispatch_id}-{job.run_id}"
        download_and_validate_bundle(
            client=client,
            dataset_name=dataset_name,
            dispatch_dir=dispatch_dir,
            run_id=job.run_id,
            validator=validator,
        )

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=dispatch_dir,
        on_task_completed=on_completed,
        poll_interval_s=float(args.poll_interval),
    )
    state.ended_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_dispatch_state(dispatch_dir, state)
    return 0


def _manifest_validator():
    """Return a callable that validates a bundle directory's manifest.

    Stub implementation: a manifest is "valid" if the file exists. Replace
    with the canonical validator from ``tools.odin.common.manifest`` when
    that exposes a public ``validate(path) -> bool`` API.
    """

    def _validate(bundle_dir: Path) -> bool:
        return (bundle_dir / "manifest.json").exists()

    return _validate


if __name__ == "__main__":
    raise SystemExit(main())
