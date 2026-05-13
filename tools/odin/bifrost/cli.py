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
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
    DispatchState,
    read_dispatch_state,
    write_dispatch_state,
)
from tools.odin.bifrost.bundle import download_and_validate_bundle
from tools.odin.bifrost.client import OsmoClient
from tools.odin.bifrost.config import BifrostConfig, BifrostConfigError, load_bifrost_config
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
    # Curated-YAML ``timeout_class``; resolved (with fallback to
    # ``cfg.default_timeout_class``) in :func:`_build_rows`. Used by
    # :func:`_bucket_and_chunk` to group rows by OSMO ``exec_timeout``.
    timeout_class: str = ""


def _load_envs_yaml(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    rows: list[dict[str, Any]] = []
    for group_rows in (data.get("groups") or {}).values():
        for env in group_rows or []:
            if env.get("keep") is True:
                rows.append(env)
    return rows


def _matches_include(task_id: str, include_glob: str | None) -> bool:
    if not include_glob:
        return True
    # Comma-separated list of globs; match if any matches (logical OR).
    return any(fnmatch.fnmatch(task_id, g.strip()) for g in include_glob.split(",") if g.strip())


def _resolve_timeout_class(env: dict[str, Any], cfg: BifrostConfig | None) -> str:
    """Pick a ``timeout_class`` for one curated env row.

    Args:
        env: The raw env dict from the curated YAML (``physx_envs.yaml``).
        cfg: The loaded bifrost config; supplies ``default_timeout_class``
            and validates the chosen class against ``timeout_classes``.
            ``None`` means the planner is running in a legacy path that
            doesn't use timeout classes; fallback returns ``""``.

    Returns:
        The class name as listed in ``cfg.timeout_classes``.

    Raises:
        BifrostConfigError: When the env's ``timeout_class`` (or the
            fallback default) is not present in ``cfg.timeout_classes``.
    """
    raw = env.get("timeout_class")
    if cfg is None or not cfg.timeout_classes:
        return str(raw) if raw is not None else ""
    chosen = str(raw) if raw else cfg.default_timeout_class
    if chosen not in cfg.timeout_classes:
        known = sorted(cfg.timeout_classes.keys())
        raise BifrostConfigError(
            f"timeout_class {chosen!r} for env {env.get('task_id')!r} is not declared in "
            f"bifrost-osmo.yaml's timeout_classes (known: {known})"
        )
    return chosen


def _build_rows(
    *,
    physx_yaml: Path,
    newton_yaml: Path | None,
    seeds: list[int],
    include_glob: str | None,
    dispatch_id: str,
    cfg: BifrostConfig | None = None,
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
            timeout_class = _resolve_timeout_class(env, cfg)
            framework_slug = framework.replace("_", "-")
            for seed in seeds:
                run_id = f"{framework_slug}_{backend}_{task_id}_{dispatch_id}_seed{seed}"
                rows.append(
                    _PlannedRow(
                        run_id=run_id,
                        task_id=task_id,
                        framework=framework,
                        backend=backend,
                        seed=seed,
                        num_envs=num_envs,
                        max_iterations=max_iter,
                        timeout_class=timeout_class,
                    )
                )
    return rows


def _planned_to_render(row: _PlannedRow) -> RenderRow:
    return RenderRow(
        run_id=row.run_id,
        osmo_task_name=osmo_safe_task_name(row.run_id),
        framework=row.framework,
        framework_runner="hugin" if row.framework in ("rsl-rl", "rsl_rl") else "munin",
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


def _planned_row_from_job(job: JobEntry) -> _PlannedRow:
    """Reconstruct a :class:`_PlannedRow` from a previously recorded :class:`JobEntry`.

    Used by the ``--retry-failed`` path so that retried jobs preserve the
    exact ``run_id`` from the parent dispatch instead of getting a fresh one.

    Args:
        job: A job entry from the parent dispatch state.

    Returns:
        A planned row carrying the parent's identifiers.
    """
    return _PlannedRow(
        run_id=job.run_id,
        task_id=job.task_id,
        framework=job.framework,
        backend=job.backend,
        seed=job.seed,
        num_envs=job.num_envs,
        max_iterations=job.max_iterations,
    )


def _find_parent_dispatch(runs_root: Path, retry_run_ids: set[str]) -> DispatchState | None:
    """Find the most recent dispatch dir whose job list is a superset of *retry_run_ids*.

    Args:
        runs_root: Root directory that contains per-dispatch subdirectories.
        retry_run_ids: Set of run_ids that must all appear in the candidate dispatch.

    Returns:
        The matching :class:`DispatchState`, or ``None`` if not found.
    """
    if not runs_root.exists():
        return None
    for d in sorted(runs_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        st = read_dispatch_state(d)
        if st is None:
            continue
        present = {j.run_id for j in st.jobs}
        if retry_run_ids <= present:
            return st
    return None


def _allocate_dispatch_id(now: dt.datetime | None = None, runs_root: Path | None = None) -> str:
    """Return a unique dispatch id string based on the current UTC timestamp.

    If *runs_root* is given and a directory with the base timestamp name already
    exists there, a monotone suffix ``-N`` is appended to avoid collisions when
    two dispatches are created within the same second (e.g. in tests).

    Args:
        now: Optional datetime override, defaults to ``dt.datetime.now(dt.UTC)``.
        runs_root: Optional parent directory; used only for collision avoidance.

    Returns:
        A string of the form ``YYYYMMDD-HHMMSS`` or ``YYYYMMDD-HHMMSS-N``.
    """
    base = (now or dt.datetime.now(dt.UTC)).strftime("%Y%m%d-%H%M%S")
    if runs_root is None or not (runs_root / base).exists():
        return base
    n = 1
    while (runs_root / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def _tail_first_running_task(
    client,
    state: DispatchState,
    dispatch_dir: Path,
    stop_event: threading.Event,
) -> None:
    """Best-effort live tail of the first task we observe RUNNING.

    Single-task by design (per spec §6.1 step 8). When that task
    terminates, we don't pick up another — the next dispatch's
    ``--verbose`` will. Best-effort: any exception is swallowed.

    Waits until the dispatch.json on disk records at least one job whose
    status has advanced past ``"pending"`` (i.e. the poll loop has seen it
    RUNNING at least once), then opens a ``follow=True`` log stream for
    that job. Reading the persisted state avoids adding extra
    ``client.status()`` calls that could disturb test counters.

    Args:
        client: An OSMO client instance with a ``logs()`` method.
        state: The current dispatch state (used to obtain the workflow id
            and to look up jobs before the first write). The on-disk
            dispatch.json is polled to detect a transitioned job.
        dispatch_dir: Root directory for this dispatch; log written to
            ``<dispatch_dir>/<run_id>/logs/osmo-tail.log``.
        stop_event: Set by the caller when polling is complete, signalling
            the thread to exit.
    """
    workflow_id = state.osmo_workflow_id
    while True:
        # Read the freshest state from disk (written by the poll loop after
        # each status check so we see the first RUNNING → "running" transition).
        # We deliberately check the disk state BEFORE testing stop_event so that
        # a very fast poll loop (e.g. poll_interval_s=0 in tests) cannot race
        # past us before we've had a chance to find a RUNNING job.
        on_disk = read_dispatch_state(dispatch_dir)
        jobs = on_disk.jobs if on_disk is not None else state.jobs
        job = next((j for j in jobs if j.status != "pending"), None)
        if job is not None:
            break
        if stop_event.wait(0.02):
            # Poll is done and we never saw a non-pending job (e.g. instant
            # complete in tests where the first disk write already shows
            # "completed" — still tail it).
            on_disk = read_dispatch_state(dispatch_dir)
            jobs = on_disk.jobs if on_disk is not None else state.jobs
            job = next((j for j in jobs if j.osmo_task_name), None)
            break
    if job is None:
        return
    log_dir = dispatch_dir / job.run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "osmo-tail.log"
    try:
        with log_path.open("ab") as fh:
            for chunk in client.logs(workflow_id, job.osmo_task_name, follow=True):
                fh.write(chunk)
                fh.flush()
                if stop_event.is_set():
                    return
    except Exception:
        pass


def _resolve_resume_dispatch(runs_root: Path, target: str) -> Path:
    if target == "LATEST":
        candidates = sorted([p for p in runs_root.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"no dispatch dirs under {runs_root}")
        return candidates[-1]
    return runs_root / target


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_bifrost_config(args.osmo_config)
    if args.pool:
        cfg = replace(cfg, pool=args.pool)
    if args.priority:
        cfg = replace(cfg, priority=args.priority)

    if args.resume:
        dispatch_dir = _resolve_resume_dispatch(args.runs_root, args.resume)
        state = read_dispatch_state(dispatch_dir)
        if state is None:
            print(f"resume target {dispatch_dir} has no dispatch.json", file=sys.stderr)
            return 2
        if state.osmo_workflow_id is None:
            print(
                f"resume target {dispatch_dir} has no osmo_workflow_id (was --dry-run only?)",
                file=sys.stderr,
            )
            return 2
        client = OsmoClient(profile=cfg.osmo_profile)
        validator = _manifest_validator()

        def on_completed(job: JobEntry) -> None:
            dataset_name = f"{cfg.bundle_dataset_prefix}-{state.dispatch_id}-{job.run_id}"
            try:
                download_and_validate_bundle(
                    client=client,
                    dataset_name=dataset_name,
                    dispatch_dir=dispatch_dir,
                    run_id=job.run_id,
                    validator=validator,
                )
            except Exception as exc:
                print(f"[bifrost] bundle download for {job.run_id} skipped: {exc}", file=sys.stderr)

        tail_stop = threading.Event()
        tail_thread: threading.Thread | None = None
        if args.verbose:
            tail_thread = threading.Thread(
                target=_tail_first_running_task,
                args=(client, state, dispatch_dir, tail_stop),
                daemon=True,
            )
            tail_thread.start()

        poll_until_terminal(
            client=client,
            state=state,
            dispatch_dir=dispatch_dir,
            on_task_completed=on_completed,
            poll_interval_s=float(args.poll_interval),
        )
        tail_stop.set()
        if tail_thread is not None:
            tail_thread.join(timeout=5)

        state.ended_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_dispatch_state(dispatch_dir, state)
        _aggregate_at_end(dispatch_dir)
        return 0

    dispatch_id = _allocate_dispatch_id(runs_root=args.runs_root)
    dispatch_dir = args.runs_root / dispatch_id
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(
        physx_yaml=args.physx_yaml,
        newton_yaml=args.newton_yaml,
        seeds=args.seeds,
        include_glob=args.include,
        dispatch_id=dispatch_id,
        cfg=cfg,
    )
    if not rows:
        print("No keep:true rows matched the include filter.", file=sys.stderr)
        return 2

    parent_dispatch_id: str | None = None
    if args.retry_failed:
        retry_run_ids = {x.strip() for x in args.retry_failed.split(",") if x.strip()}
        parent = _find_parent_dispatch(args.runs_root, retry_run_ids)
        if parent is None:
            print(
                f"no recent dispatch contains all run_ids {sorted(retry_run_ids)}",
                file=sys.stderr,
            )
            return 2
        parent_dispatch_id = parent.dispatch_id
        rows = [_planned_row_from_job(j) for j in parent.jobs if j.run_id in retry_run_ids]
        if not rows:
            print(
                "retry-failed run_ids did not match any rows in parent dispatch",
                file=sys.stderr,
            )
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
        parent_dispatch_id=parent_dispatch_id,
    )
    write_dispatch_state(dispatch_dir, state)

    if args.dry_run:
        print(f"[dry-run] wrote {workflow_yaml_path}")
        return 0

    client = OsmoClient(profile=cfg.osmo_profile)
    rsync_pairs: list[tuple[str, str]] = []
    if args.rsync:
        rsync_pairs.append((cfg.code_delivery.source_root, "/workspace/IsaacLab/" + cfg.code_delivery.source_root))
    workflow_id = client.submit(workflow_yaml_path, rsync_pairs=rsync_pairs, pool=cfg.pool)
    state.osmo_workflow_id = workflow_id
    write_dispatch_state(dispatch_dir, state)

    validator = _manifest_validator()

    def on_completed(job: JobEntry) -> None:
        dataset_name = f"{cfg.bundle_dataset_prefix}-{dispatch_id}-{job.run_id}"
        try:
            download_and_validate_bundle(
                client=client,
                dataset_name=dataset_name,
                dispatch_dir=dispatch_dir,
                run_id=job.run_id,
                validator=validator,
            )
        except Exception as exc:
            # Bundle download failure (missing DATA credential, dataset
            # absent because outputs: was stripped from the workflow, etc.)
            # should not abort the poller -- the dispatch state machine
            # still wants to mark the job completed and keep tracking the
            # remaining tasks. The operator can re-download once the
            # credential lands.
            print(f"[bifrost] bundle download for {job.run_id} skipped: {exc}", file=sys.stderr)

    tail_stop = threading.Event()
    tail_thread: threading.Thread | None = None
    if args.verbose:
        tail_thread = threading.Thread(
            target=_tail_first_running_task,
            args=(client, state, dispatch_dir, tail_stop),
            daemon=True,
        )
        tail_thread.start()

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=dispatch_dir,
        on_task_completed=on_completed,
        poll_interval_s=float(args.poll_interval),
    )
    tail_stop.set()
    if tail_thread is not None:
        tail_thread.join(timeout=5)

    state.ended_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_dispatch_state(dispatch_dir, state)
    _aggregate_at_end(dispatch_dir)
    return 0


def _aggregate_at_end(dispatch_dir: Path) -> None:
    """Run valhalla aggregate + write aggregate.json. Best-effort.

    Mirrors asgard's end-of-dispatch hook so Bifrost dispatches land with
    the same aggregate artifact (used by the dashboard's compare-runs
    feature and by downstream tooling). Failures are non-fatal: bundle
    data on disk is independently useful even if the aggregator can't
    compose it.
    """
    try:
        from tools.odin.valhalla import aggregate_dispatch, write_aggregate

        agg = aggregate_dispatch(dispatch_dir)
        write_aggregate(dispatch_dir, agg)
        print(f"[bifrost] aggregate.json written ({agg.get('totals', {})})")
    except Exception as exc:
        print(f"[bifrost] aggregate skipped: {exc}", file=sys.stderr)


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
