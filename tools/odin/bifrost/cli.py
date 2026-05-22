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

from tools.odin.asgard.budgets import Budgets, load_budgets
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
    p.add_argument(
        "--budgets-yaml",
        type=Path,
        default=_DEFAULT_BUDGETS_YAML,
        help=(
            "Per-task wall-clock budget table. Defaults to "
            "tools/odin/config/job_budgets.yaml. Bifrost looks up "
            "(task_id, framework) here to size each OSMO workflow's "
            "exec_timeout (max across the chunk)."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--keep-osmo-datasets",
        action="store_true",
        help=(
            "Skip deleting each OSMO dataset after its bundle is "
            "downloaded + validated locally. By default Bifrost "
            "deletes the OSMO copy to reclaim bucket storage; pass "
            "this flag if you want to keep the datasets (e.g. for "
            "side-by-side comparison or re-download)."
        ),
    )
    return p


_DEFAULT_BUDGETS_YAML = Path("tools/odin/config/job_budgets.yaml")


@dataclass(frozen=True)
class _PlannedRow:
    run_id: str
    task_id: str
    framework: str  # rsl-rl | skrl
    backend: str  # physx | newton
    seed: int
    num_envs: int
    max_iterations: int
    # Per-task wall-clock budget in seconds, resolved from
    # ``tools/odin/config/job_budgets.yaml`` via
    # :func:`tools.odin.asgard.budgets.load_budgets`. Falls back to
    # ``budgets.defaults[framework]`` when the task isn't listed.
    # :func:`_bucket_and_chunk` sorts on this field and emits one OSMO
    # workflow per chunk with ``exec_timeout`` = max-of-chunk.
    per_task_timeout_s: int = 0


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


def _gpu_class_from_platform(platform: str) -> str:
    """Map an OSMO ``resources.platform`` string to a ``gpu_multipliers`` key.

    OSMO platform names follow ``<chassis>-<gpu>`` (``ovx-l40s``,
    ``dgx-h100``, ``ovx-l40``). The ``gpu_multipliers`` table in
    ``job_budgets.yaml`` keys on the bare GPU portion (``l40s``,
    ``h100``, ``l40``), so we strip a known chassis prefix.

    Args:
        platform: ``cfg.defaults.resources.platform`` from
            ``bifrost-osmo.yaml`` (e.g. ``"ovx-l40s"``).

    Returns:
        The GPU class key to look up in ``budgets.gpu_multipliers``.
        Returns ``platform`` unchanged when no known prefix matches —
        the caller falls back to ``gpu_multipliers.default``.
    """
    for prefix in ("ovx-", "dgx-"):
        if platform.startswith(prefix):
            return platform[len(prefix) :]
    return platform


def _resolve_per_task_timeout_s(
    task_id: str,
    framework: str,
    budgets: Budgets,
    gpu_class: str | None = None,
) -> int:
    """Look up the per-task wall-clock budget in seconds.

    Reuses :class:`tools.odin.asgard.budgets.Budgets` so Bifrost and
    Asgard share one source of truth (``tools/odin/config/job_budgets.yaml``).
    When ``gpu_class`` is provided, the matching multiplier from
    ``budgets.gpu_multipliers`` is applied (falling back to
    ``gpu_multipliers.default`` when the class is unknown).

    Framework keys are normalised to the underscored spelling
    (``"rsl_rl"``, ``"skrl"``) before lookup so curated YAMLs that say
    ``framework: rsl-rl`` resolve against the same table.

    Args:
        task_id: Gym task id (e.g. ``Isaac-Ant-Direct-v0``).
        framework: ``"rsl_rl"`` / ``"rsl-rl"`` or ``"skrl"``.
        budgets: Loaded :class:`Budgets` table.
        gpu_class: GPU class for the OSMO pool (e.g. ``"l40s"``,
            ``"h100"``); pass ``None`` to skip multipliers entirely.

    Returns:
        Whole seconds of wall-clock budget, with the GPU multiplier
        applied when provided.

    Raises:
        BifrostConfigError: When the task is missing AND
            ``budgets.defaults[framework]`` is absent — bail out at plan
            time instead of emitting a workflow with a placeholder
            timeout.
    """
    fw_key = framework.replace("-", "_")
    per_task = budgets.budgets.get(task_id, {})
    base = per_task.get(fw_key)
    if base is None:
        base = budgets.defaults.get(fw_key)
    if base is None:
        raise BifrostConfigError(
            f"job_budgets.yaml has no entry for task {task_id!r} under framework "
            f"{fw_key!r} and no defaults.{fw_key} fallback; add either an entry in "
            f"budgets: or a defaults.{fw_key}: value"
        )
    multiplier = 1.0
    if gpu_class is not None:
        mults = budgets.gpu_multipliers or {}
        multiplier = float(mults.get(gpu_class, mults.get("default", 1.0)))
    return int(base * multiplier)


def _build_rows(
    *,
    physx_yaml: Path,
    newton_yaml: Path | None,
    seeds: list[int],
    include_glob: str | None,
    dispatch_id: str,
    cfg: BifrostConfig | None = None,
    budgets: Budgets,
) -> list[_PlannedRow]:
    # Derive the GPU class from the pool's platform string so per-task
    # budgets get scaled by the right gpu_multiplier entry (e.g.
    # ``ovx-l40s`` -> ``l40s`` -> ``budgets.gpu_multipliers["l40s"]``).
    gpu_class = None
    if cfg is not None and cfg.defaults.resources.platform:
        gpu_class = _gpu_class_from_platform(cfg.defaults.resources.platform)
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
            per_task_timeout_s = _resolve_per_task_timeout_s(task_id, framework, budgets, gpu_class)
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
                        per_task_timeout_s=per_task_timeout_s,
                    )
                )
    return rows


def _bucket_and_chunk(
    rows: list[_PlannedRow], chunk_size: int, *, exec_timeout_s: int
) -> list[tuple[int, int, list[_PlannedRow]]]:
    """Chunk rows by ``chunk_size`` and apply a fixed ``exec_timeout_s`` per chunk.

    Pure function: no I/O, no logging, no exceptions on empty input.
    Each output tuple is ``(chunk_index, exec_timeout_s, rows)``:

    - ``chunk_index`` is the 0-based position of the chunk.
    - ``exec_timeout_s`` is the OSMO workflow ``exec_timeout`` in seconds.
      Since each task is wrapped in its own group and the OSMO clock is
      per-group, a generous fixed value works for all task sizes: short
      tasks finish and exit early, long tasks have headroom.
    - ``rows`` has at most ``chunk_size`` entries.

    Sort key is ``(per_task_timeout_s, task_id, backend, seed)`` — ties
    resolve deterministically so reruns and ``--resume`` see the same
    layout.

    Args:
        rows: Planned rows from :func:`_build_rows`.
        chunk_size: Maximum rows per OSMO workflow. The planner uses
            ``cfg.chunk_size``.
        exec_timeout_s: Fixed workflow ``exec_timeout`` in seconds,
            sourced from ``cfg.defaults.exec_timeout``.

    Returns:
        A list of ``(chunk_index, exec_timeout_s, chunk_rows)`` tuples,
        one per OSMO workflow that will be submitted.
    """
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: (r.per_task_timeout_s, r.task_id, r.backend, r.seed))
    out: list[tuple[int, int, list[_PlannedRow]]] = []
    for idx, start in enumerate(range(0, len(sorted_rows), chunk_size)):
        chunk = sorted_rows[start : start + chunk_size]
        out.append((idx, exec_timeout_s, chunk))
    return out


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


def _planned_row_from_job(
    job: JobEntry,
    budgets: Budgets,
    gpu_class: str | None = None,
) -> _PlannedRow:
    """Reconstruct a :class:`_PlannedRow` from a previously recorded :class:`JobEntry`.

    Used by the ``--retry-failed`` path so that retried jobs preserve the
    exact ``run_id`` from the parent dispatch instead of getting a fresh
    one. Re-resolves the per-task timeout against the current budgets
    table so a refreshed ``job_budgets.yaml`` (e.g. operator bumped a
    task that consistently times out) is picked up on retry.

    Args:
        job: A job entry from the parent dispatch state.
        budgets: Current :class:`Budgets` table.
        gpu_class: GPU class for the pool the retry will target;
            forwarded to :func:`_resolve_per_task_timeout_s` so the
            same gpu_multiplier the forward path applied at planning
            is re-applied here. ``None`` skips multiplier scaling.

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
        per_task_timeout_s=_resolve_per_task_timeout_s(job.task_id, job.framework, budgets, gpu_class),
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
        if not state.osmo_workflow_ids and state.osmo_workflow_id is None:
            print(
                f"resume target {dispatch_dir} has no OSMO workflow ids (was --dry-run only?)",
                file=sys.stderr,
            )
            return 2
        client = OsmoClient(profile=cfg.osmo_profile)
        validator = _manifest_validator()

        def on_completed(job: JobEntry) -> None:
            dataset_name = f"{cfg.bundle_dataset_prefix}-{state.dispatch_id}-{job.run_id}"
            try:
                result = download_and_validate_bundle(
                    client=client,
                    dataset_name=dataset_name,
                    dispatch_dir=dispatch_dir,
                    run_id=job.run_id,
                    validator=validator,
                )
            except Exception as exc:
                print(f"[bifrost] bundle download for {job.run_id} skipped: {exc}", file=sys.stderr)
                return
            if result.is_valid and not args.keep_osmo_datasets:
                try:
                    client.dataset_delete(dataset_name)
                except Exception as exc:
                    print(
                        f"[bifrost] dataset cleanup for {dataset_name} skipped: {exc}",
                        file=sys.stderr,
                    )

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

    budgets = load_budgets(args.budgets_yaml)

    rows = _build_rows(
        physx_yaml=args.physx_yaml,
        newton_yaml=args.newton_yaml,
        seeds=args.seeds,
        include_glob=args.include,
        dispatch_id=dispatch_id,
        cfg=cfg,
        budgets=budgets,
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
        retry_gpu_class = (
            _gpu_class_from_platform(cfg.defaults.resources.platform) if cfg.defaults.resources.platform else None
        )
        rows = [_planned_row_from_job(j, budgets, retry_gpu_class) for j in parent.jobs if j.run_id in retry_run_ids]
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

    # One OSMO workflow per chunk. exec_timeout is a fixed global
    # ceiling (``cfg.defaults.exec_timeout``) rather than a per-chunk
    # max because each task is wrapped in its own group — the OSMO
    # clock is per-group, so short tasks finish and exit on their own
    # while a generous ceiling catches genuine wedges.
    buckets = _bucket_and_chunk(rows, cfg.chunk_size, exec_timeout_s=cfg.defaults.exec_timeout)
    chunk_render: list[tuple[int, int, list[_PlannedRow], Path]] = []
    for idx, exec_timeout_s, chunk_rows in buckets:
        yaml_path = dispatch_dir / f"workflow.{idx}.yaml"
        yaml_body = render_workflow_yaml(
            dispatch_id=dispatch_id,
            rows=[_planned_to_render(r) for r in chunk_rows],
            cfg=cfg,
            tarball_path=tarball_path,
            exec_timeout=f"{exec_timeout_s}s",
        )
        yaml_path.write_text(yaml_body)
        chunk_render.append((idx, exec_timeout_s, chunk_rows, yaml_path))

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
        print(f"[dry-run] wrote {len(buckets)} workflow YAML file(s) under {dispatch_dir}")
        return 0

    client = OsmoClient(profile=cfg.osmo_profile)
    rsync_pairs: list[tuple[str, str]] = []
    if args.rsync:
        rsync_pairs.append((cfg.code_delivery.source_root, "/workspace/IsaacLab/" + cfg.code_delivery.source_root))
    # Submit one workflow per chunk. Persist the dispatch.json after each
    # submit so a failure mid-way leaves the prior workflows' ids on
    # disk and the resume path can re-attach.
    for _idx, _max_s, _chunk_rows, yaml_path in chunk_render:
        wf_id = client.submit(yaml_path, rsync_pairs=rsync_pairs, pool=cfg.pool)
        state.osmo_workflow_ids.append(wf_id)
        write_dispatch_state(dispatch_dir, state)

    validator = _manifest_validator()

    def on_completed(job: JobEntry) -> None:
        dataset_name = f"{cfg.bundle_dataset_prefix}-{dispatch_id}-{job.run_id}"
        try:
            result = download_and_validate_bundle(
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
            return
        # Default: reclaim OSMO bucket storage by deleting the dataset
        # now that we have the bundle on local disk. --keep-osmo-datasets
        # disables this for operators who want to keep the OSMO copy.
        if result.is_valid and not args.keep_osmo_datasets:
            try:
                client.dataset_delete(dataset_name)
            except Exception as exc:
                print(
                    f"[bifrost] dataset cleanup for {dataset_name} skipped: {exc}",
                    file=sys.stderr,
                )

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
