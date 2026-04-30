# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""run_dispatch — top-level orchestration for an Asgard dispatch run."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.jobs import FailureInfo, JobEntry, SkippedEntry, build_queue_from_env_lists
from tools.odin.asgard.cleanup import sweep_orphan_trainers
from tools.odin.asgard.preflight import preflight_valkyrie
from tools.odin.asgard.provisioner import provision_valkyrie
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
    DispatchState,
    FleetSnapshot,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)
from tools.odin.asgard.transport import RsyncRunner, ShellRsyncRunner, ShellSSHRunner, SSHRunner
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions
from tools.odin.valhalla.dashboard.retry_db import RetryDB

__all__ = ["DispatchOptions", "resolve_dispatch_dir", "run_dispatch"]

# Minimum CUDA toolkit version required for Newton (warp) workloads.
_NEWTON_CUDA_FLOOR: tuple[int, int] = (12, 4)


@dataclass
class DispatchOptions:
    """Options controlling one dispatch run.

    Args:
        seeds: RNG seeds to expand each env-list row across.
        max_infrastructure_retries: Per-job infrastructure retry cap passed to
            :class:`~tools.odin.asgard.worker.ValkyrieWorker`.
        per_job_timeout_s: Wall-clock timeout [s] per job.
        fresh: When ``True``, wipe and re-provision every host before running.
        skip_preflight: When ``True``, continue even if some hosts fail
            preflight (healthy-only dispatch).
        include_filter: Optional fnmatch patterns on ``task_id``; rows not
            matching any pattern are skipped.
        verbose: Print per-job completion lines to stdout.
        retry_failed: Explicit list of ``run_id`` values to re-attempt on a
            resume even though they are ``"failed"`` in the prior state.
        retry_all_failed: When ``True`` on a resume, flip every prior
            ``failed`` job back to ``pending`` (regardless of failure kind).
            Mutually exclusive with ``retry_failed``.
        skip_aggregate: When ``True``, skip the automatic
            :func:`~tools.odin.valhalla.aggregate_dispatch` + write at the
            end of :func:`run_dispatch`. Default ``False``.
        consecutive_failure_quarantine: Number of consecutive per-worker
            failures that trigger host quarantine (``host_down`` +
            worker exit). ``0`` disables the circuit-breaker. Default
            ``3``.
        preflight_auto_restart: When ``True`` (default), automatically
            restart the container and re-probe on NVML wedge during
            preflight. Pass ``False`` to preserve strict-failure semantics.
        detached_mode: When ``True`` (default), workers use the
            submit-and-poll model so a transient network blip between
            dispatcher and Valkyrie no longer kills in-flight training.
            Set to ``False`` (via ``--legacy-pty-mode``) to fall back to
            the per-job PTY-tied SSH path during initial rollout.
        poll_interval_s: Seconds between poll ticks in detached mode.
        live_retry_poll_s: Poll period [s] for live retry rows in the
            retry DB while this dispatch still has active work.
    """

    seeds: list[int]
    max_infrastructure_retries: int = 2
    per_job_timeout_s: int = 43200
    fresh: bool = False
    skip_preflight: bool = False
    include_filter: list[str] | None = None
    verbose: bool = False
    retry_failed: list[str] | None = None
    retry_all_failed: bool = False
    skip_aggregate: bool = False
    consecutive_failure_quarantine: int = 3
    preflight_auto_restart: bool = True
    detached_mode: bool = True
    poll_interval_s: float = 30.0
    live_retry_poll_s: float = 5.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dispatch_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def resolve_dispatch_dir(runs_root: Path, resume: str | None) -> Path:
    """Return the dispatch directory (creating one if ``resume is None``).

    - ``resume=None``: create a fresh ``runs_root/<UTC-now>/`` directory.
    - ``resume="LATEST"``: return the most-recent existing subdirectory.
    - ``resume="<dispatch_id>"``: return ``runs_root/<dispatch_id>/`` (must exist).

    Args:
        runs_root: Parent directory that holds dispatch subdirectories.
        resume: Resume mode selector. ``None`` for a fresh dispatch,
            ``"LATEST"`` for the newest existing directory, or an exact
            dispatch-id string.

    Returns:
        Resolved :class:`~pathlib.Path` to the dispatch directory.

    Raises:
        FileNotFoundError: When ``resume="LATEST"`` but no subdirectories
            exist, or when ``resume=<id>`` but the named directory is absent.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    if resume is None:
        dispatch_dir = runs_root / _dispatch_id_now()
        dispatch_dir.mkdir(parents=True, exist_ok=True)
        return dispatch_dir
    if resume == "LATEST":
        children = sorted(p for p in runs_root.iterdir() if p.is_dir())
        if not children:
            raise FileNotFoundError(f"No prior dispatch directories under {runs_root}")
        return children[-1]
    candidate = runs_root / resume
    if not candidate.exists():
        raise FileNotFoundError(f"Resume target {candidate} does not exist")
    return candidate


def _snapshot_fleet_yaml(fleet: Fleet, dispatch_dir: Path) -> None:
    """Write fleet.yaml.snapshot alongside dispatch.json for audit."""
    payload = {
        "fleet_name": fleet.fleet_name,
        "hosts": [
            {
                "host": h.host,
                "ssh_user": h.ssh_user,
                "ssh_key": str(h.ssh_key) if h.ssh_key is not None else None,
                "isaaclab_path": h.isaaclab_path,
                "container_name": h.container_name,
                "labels": h.labels,
            }
            for h in fleet.hosts
        ],
    }
    (dispatch_dir / "fleet.yaml.snapshot").write_text(json.dumps(payload, indent=2))


def _write_preflight(results, dispatch_dir: Path, dispatch_id: str) -> None:
    payload = {
        "schema_version": "1.0",
        "dispatch_id": dispatch_id,
        "checked_at": _utc_now_iso(),
        "hosts": [
            {
                "host": r.host,
                "ok": r.ok,
                "checks": r.checks,
                "message": r.message,
                "recovery_attempted": r.recovery_attempted,
                "recovery_succeeded": r.recovery_succeeded,
            }
            for r in results
        ],
    }
    (dispatch_dir / "preflight.json").write_text(json.dumps(payload, indent=2))


def _apply_state_event(
    state: DispatchState,
    ev: StateEvent,
    jobs_by_id: dict[str, JobEntry] | None = None,
    verbose: bool = False,
) -> int:
    """Apply one worker state event to the shared :class:`DispatchState`.

    Mutates ``state`` in place: flips job status, records host assignment,
    and updates the matching :class:`FleetSnapshot`.

    Args:
        state: Shared dispatch state to mutate.
        ev: State event emitted by a :class:`ValkyrieWorker`.
        jobs_by_id: ``run_id → JobEntry`` lookup rebuilt by the caller.
            When ``None``, falls back to scanning ``state.jobs``; this
            keeps unit tests that exercise host-only transitions
            (``recovered`` / ``host_down``) ergonomic.
        verbose: When ``True``, print per-job completion/failure lines.

    Returns:
        ``1`` when the event advances the "remaining" counter
        (``completed`` or ``failed``), ``0`` otherwise.
    """
    if jobs_by_id is None:
        jobs_by_id = {jj.run_id: jj for jj in state.jobs}
    j = jobs_by_id.get(ev.run_id)
    if ev.transition == "running":
        if j is not None:
            j.status = "running"
            j.started_at = ev.started_at
            j.assigned_to = ev.host
        for f in state.fleet:
            if f.host == ev.host:
                f.status = "busy"
                f.current_run_id = ev.run_id
        return 0
    if ev.transition == "completed":
        if j is not None:
            j.status = "completed"
            j.ended_at = ev.ended_at
        for f in state.fleet:
            if f.host == ev.host:
                f.status = "idle"
                f.current_run_id = None
        if verbose and j is not None:
            print(f"[{_utc_now_iso()}] COMPLETE {j.run_id} on {ev.host}")
        return 1
    if ev.transition == "failed":
        if j is not None:
            j.status = "failed"
            j.failure = ev.failure
            j.ended_at = ev.ended_at
        for f in state.fleet:
            if f.host == ev.host:
                f.status = "idle"
                f.current_run_id = None
                if ev.failure is not None:
                    f.last_error = ev.failure.message
        if verbose and j is not None:
            kind = ev.failure.kind if ev.failure else "unknown"
            print(f"[{_utc_now_iso()}] FAIL     {j.run_id} on {ev.host} (kind={kind})")
        return 1
    if ev.transition == "recovered":
        for f in state.fleet:
            if f.host == ev.host:
                f.last_error = "gpu_lost: recovered"
        return 0
    if ev.transition == "host_down":
        kind = ev.failure.kind if ev.failure is not None else "unknown"
        detail = ev.failure.message if ev.failure is not None else "unknown"
        for f in state.fleet:
            if f.host == ev.host:
                f.status = "down"
                f.last_error = f"{kind}: {detail}"
                f.current_run_id = None  # worker re-queued the job (or quarantined for circuit_breaker)
        # Worker re-queued the in-flight job; reset its dispatch.json row from
        # 'running' back to 'pending' so it (a) is eligible to be picked up by
        # another healthy worker, and (b) is caught by the post-dispatch sweep
        # if no host remains. Without this reset the job stays as 'running'
        # forever in the final state.
        for j in state.jobs:
            if j.run_id == ev.run_id and j.status == "running":
                j.status = "pending"
                j.assigned_to = None
                j.started_at = None
        if ev.failure is not None and ev.failure.kind == "circuit_breaker":
            from tools.odin.asgard.state import QuarantinedHost

            state.quarantined_hosts.append(
                QuarantinedHost(
                    host=ev.host,
                    reason=ev.failure.kind,
                    last_run_id=ev.run_id,
                    at=_utc_now_iso(),
                )
            )
        return 0
    return 0


def _sweep_pending_after_dispatch(state: DispatchState) -> None:
    """Terminal-fail pending jobs left behind when no healthy host remains.

    Called after :func:`run_dispatch`'s main loop exits.  Pending jobs at
    that point either (a) were re-queued from a ``host_down`` and never
    picked up by another worker because every other host was already down,
    or (b) fell through the sentinel race (see ``worker.py``).  Both cases:
    no host can run them.  Mark each ``failed`` with ``kind="gpu_lost"``
    so the dispatch report and aggregator account for them.

    Only sweeps when at least one host is marked ``down``.  Pending jobs in
    a fully-healthy fleet point to a different bug and should remain
    visible (operator can ``--resume`` and investigate).
    """
    if not any(f.status == "down" for f in state.fleet):
        return
    for j in state.jobs:
        if j.status == "pending":
            j.status = "failed"
            j.failure = FailureInfo(
                kind="gpu_lost",
                message="no healthy host available; all hosts marked down",
                details={"preferred_not": sorted(j.preferred_not)},
            )
            j.ended_at = _utc_now_iso()


def _apply_retry_options(jobs: list[JobEntry], options: DispatchOptions) -> None:
    """Resume-time retry logic. Mutates ``jobs`` in place.

    ``options.retry_failed``: explicit run_id list → flip matching failed
    jobs to pending (preserving attempts / timestamps; this is a targeted
    poke).

    ``options.retry_all_failed``: sweep every failed job back to pending,
    clearing attempts/assignment/timestamps so the fresh attempt looks
    like a first try. Convenient when a host-level wedge produced N>>1
    spurious failures and enumerating run_ids by hand is impractical.

    The two are mutually exclusive at the CLI surface; this function
    short-circuits on the targeted form first.
    """
    if options.retry_failed:
        retry_set = set(options.retry_failed)
        for j in jobs:
            if j.run_id in retry_set and j.status == "failed":
                j.status = "pending"
                j.failure = None
        return
    if options.retry_all_failed:
        for j in jobs:
            if j.status == "failed":
                j.status = "pending"
                j.failure = None
                j.attempts = 0
                j.assigned_to = None
                j.started_at = None
                j.ended_at = None


def _consume_live_retries(
    *,
    retry_db: RetryDB,
    dispatch_id: str,
    jobs_by_id: dict[str, JobEntry],
    job_q: queue.Queue,
    live_retry_run_ids: set[str],
) -> int:
    """Requeue failed jobs that were tagged for retry while this dispatch is live."""
    added = 0
    for run_id in sorted(retry_db.read_pending(dispatch_id)):
        if run_id in live_retry_run_ids:
            continue
        job = jobs_by_id.get(run_id)
        if job is None or job.status != "failed":
            continue
        job.status = "pending"
        job.failure = None
        job.assigned_to = None
        job.started_at = None
        job.ended_at = None
        job_q.put(job)
        live_retry_run_ids.add(run_id)
        added += 1
    return added


def _mark_live_retry_consumed(
    *,
    retry_db: RetryDB,
    dispatch_id: str,
    ev: StateEvent,
    live_retry_run_ids: set[str],
) -> None:
    """Mark a terminal live retry event consumed in the retry DB."""
    if ev.run_id not in live_retry_run_ids or ev.transition not in {"completed", "failed"}:
        return
    outcome = "completed" if ev.transition == "completed" else "failed"
    failure_kind = ev.failure.kind if ev.failure is not None else None
    try:
        retry_db.mark_consumed(
            dispatch_id,
            ev.run_id,
            retry_dispatch_id=dispatch_id,
            outcome=outcome,
            failure_kind=failure_kind,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] failed to mark retry consumed for {ev.run_id}: {exc}")
    finally:
        live_retry_run_ids.discard(ev.run_id)


def _merge_jobs(existing: list[JobEntry], fresh: list[JobEntry]) -> list[JobEntry]:
    """Preserve completed / failed from existing; take pending / running /
    assigned flipped-to-pending rows from existing too; new rows from fresh.

    Raises ValueError if the fresh list is not a superset of the existing list
    (dispatch_id / seeds mismatch → user should start a new dispatch).
    """
    by_id = {j.run_id: j for j in existing}
    merged: list[JobEntry] = []
    fresh_ids: set[str] = set()
    for f in fresh:
        fresh_ids.add(f.run_id)
        if f.run_id in by_id:
            merged.append(by_id[f.run_id])
        else:
            merged.append(f)
    missing = set(by_id) - fresh_ids
    if missing:
        raise ValueError(
            f"Resume target contains jobs not in the current queue: {sorted(missing)[:3]}... "
            f"(dispatch_id / seeds / include_filter changed; start a new dispatch)"
        )
    return merged


def run_dispatch(
    fleet: Fleet,
    physx_yaml: Path | None,
    newton_yaml: Path | None,
    dispatch_dir: Path,
    options: DispatchOptions,
    *,
    ssh: SSHRunner | None = None,
    rsync: RsyncRunner | None = None,
) -> DispatchState:
    """Orchestrate one distributed dispatch.

    Inputs:
      - ``fleet``: Valkyrie host list.
      - ``physx_yaml`` / ``newton_yaml``: curated T2.1 env lists (at least one).
      - ``dispatch_dir``: ``odin_runs/<dispatch_id>/`` directory (created by
        :func:`resolve_dispatch_dir`; either fresh or a resume target).
      - ``options``: seeds, timeouts, etc. See :class:`DispatchOptions`.
      - ``ssh`` / ``rsync``: transport injection points. Default to
        ``ShellSSHRunner`` / ``ShellRsyncRunner``.

    Args:
        fleet: Valkyrie host list.
        physx_yaml: Path to the PhysX env list YAML, or ``None`` to skip.
        newton_yaml: Path to the Newton env list YAML, or ``None`` to skip.
        dispatch_dir: Dispatch directory, typically created by
            :func:`resolve_dispatch_dir`.
        options: Dispatch options (seeds, timeouts, filters).
        ssh: SSH runner; defaults to :class:`~tools.odin.asgard.transport.ShellSSHRunner`.
        rsync: Rsync runner; defaults to
            :class:`~tools.odin.asgard.transport.ShellRsyncRunner`.

    Returns:
        Final :class:`~tools.odin.asgard.state.DispatchState` (also written to
        ``<dispatch_dir>/dispatch.json``).

    Raises:
        RuntimeError: When preflight fails for all hosts (or any host when
            ``options.skip_preflight`` is ``False``).
    """
    ssh = ssh or ShellSSHRunner()
    rsync = rsync or ShellRsyncRunner()
    dispatch_id = dispatch_dir.name

    fresh_jobs, fresh_skipped = build_queue_from_env_lists(
        physx_yaml=physx_yaml,
        newton_yaml=newton_yaml,
        seeds=options.seeds,
        dispatch_id=dispatch_id,
        include_filter=options.include_filter,
    )

    # Load prior state for resume if it exists.
    prior_state = read_dispatch_state(dispatch_dir)
    if prior_state is not None:
        # Reconcile orphans (PR4 / punch-list #7b): for any 'running' job,
        # check the remote for a completed manifest, alive process, or
        # neither, and mutate accordingly. Must run BEFORE
        # reset_in_flight_to_pending — that function would otherwise lose
        # the assigned_to we need to find the host.
        from tools.odin.asgard.reconcile import reconcile_orphans

        reconcile_orphans(
            fleet=fleet,
            jobs=prior_state.jobs,
            dispatch_dir=dispatch_dir,
            ssh=ssh,
            rsync=rsync,
            detached_mode=options.detached_mode,
        )
        # Flip remaining in-flight (those still in 'running' after reconcile,
        # e.g. assigned_to=None edge cases) → pending.
        reset_in_flight_to_pending(prior_state)
        merged_jobs = _merge_jobs(prior_state.jobs, fresh_jobs)
        # Resume preserves the prior skipped[] verbatim; we don't re-evaluate.
        merged_skipped = list(prior_state.skipped)
        started_at = prior_state.started_at
        _apply_retry_options(merged_jobs, options)
    else:
        merged_jobs = fresh_jobs
        merged_skipped = fresh_skipped
        started_at = _utc_now_iso()

    # Pre-dispatch summary of skipped (task, backend) pairs. One block per
    # (task_id, backend, reason) combination, with all affected seeds collapsed.
    if merged_skipped:
        from collections import defaultdict

        grouped: dict[tuple[str, str, str], list[SkippedEntry]] = defaultdict(list)
        for sk in merged_skipped:
            grouped[(sk.task_id, sk.backend, sk.reason)].append(sk)
        print(f"[INFO] Skipping {len(merged_skipped)} (task, backend) pairs:")
        for (task_id, backend, reason), rows in sorted(grouped.items()):
            seeds_str = ", ".join(str(r.seed) for r in sorted(rows, key=lambda r: r.seed))
            if reason == "native_backend_mismatch":
                native = rows[0].native_backend
                detail = f"native: {native}"
            else:
                avail = rows[0].presets_available
                detail = f"available: {avail}"
            print(f"[INFO]   {task_id} × {backend} (seeds {seeds_str}) — {reason} ({detail})")

    # Snapshot fleet.yaml.
    _snapshot_fleet_yaml(fleet, dispatch_dir)

    # Preflight.
    pre_results = [
        preflight_valkyrie(
            h,
            ssh=ssh,
            auto_restart=options.preflight_auto_restart,
            newton_cuda_floor=_NEWTON_CUDA_FLOOR,
        )
        for h in fleet.hosts
    ]
    _write_preflight(pre_results, dispatch_dir, dispatch_id)

    healthy: list[ValkyrieConfig] = []
    down_hosts: set[str] = set()
    for host, res in zip(fleet.hosts, pre_results):
        if res.ok:
            healthy.append(host)
        else:
            down_hosts.add(host.host)

    if not healthy:
        # Emit a final dispatch.json before raising so the audit record is complete.
        state = DispatchState(
            schema_version=SCHEMA_VERSION,
            dispatch_id=dispatch_id,
            started_at=started_at,
            ended_at=_utc_now_iso(),
            seeds=options.seeds,
            commit_sha="",
            fleet=[
                FleetSnapshot(
                    host=h.host,
                    status="down",
                    last_error=next((r.message for r in pre_results if r.host == h.host), None),
                )
                for h in fleet.hosts
            ],
            jobs=merged_jobs,
            skipped=merged_skipped,
        )
        write_dispatch_state(dispatch_dir, state)
        raise RuntimeError(f"preflight failed for all {len(fleet.hosts)} hosts; see preflight.json")

    if down_hosts and not options.skip_preflight:
        state = DispatchState(
            schema_version=SCHEMA_VERSION,
            dispatch_id=dispatch_id,
            started_at=started_at,
            ended_at=_utc_now_iso(),
            seeds=options.seeds,
            commit_sha="",
            fleet=[
                FleetSnapshot(
                    host=h.host,
                    status="down" if h.host in down_hosts else "idle",
                    last_error=next((r.message for r in pre_results if r.host == h.host and not r.ok), None),
                )
                for h in fleet.hosts
            ],
            jobs=merged_jobs,
            skipped=merged_skipped,
        )
        write_dispatch_state(dispatch_dir, state)
        raise RuntimeError(
            f"preflight failed for {len(down_hosts)}/{len(fleet.hosts)} hosts; "
            f"pass --skip-preflight to run on healthy ones only. See preflight.json."
        )

    # Provision every healthy host to the controller's working tree.
    working_tree = Path.cwd()  # caller invokes from the repo root
    commit_sha = ""
    for host in healthy:
        pr = provision_valkyrie(host, working_tree, fresh=options.fresh, ssh=ssh, rsync=rsync)
        if not pr.ok:
            down_hosts.add(host.host)
        else:
            commit_sha = commit_sha or pr.commit_sha
    healthy = [h for h in healthy if h.host not in down_hosts]

    # Sweep orphan trainers left behind by a prior dispatch (legacy-PTY mode
    # could leak ``docker exec``-launched trainers when the dispatcher's SSH
    # was killed mid-run). Detached mode does not create orphans, but a fleet
    # may have run a legacy dispatch yesterday — sweep regardless.
    for host in healthy:
        sweep = sweep_orphan_trainers(host, ssh=ssh)
        if sweep.killed_count > 0:
            print(f"[INFO] {host.host}: killed {sweep.killed_count} orphan trainer(s)")
        elif not sweep.ok:
            print(f"[WARN] {host.host}: zombie sweep failed: {sweep.message}")

    # Filter Newton jobs when no host meets the CUDA floor. Mark them failed
    # with a clear kind so the dispatch report (and operator) sees the gap.
    healthy_hosts_set = {h.host for h in healthy}
    newton_capable_hosts = [r.host for r in pre_results if r.ok and r.newton_available and r.host in healthy_hosts_set]
    if not newton_capable_hosts:
        for j in merged_jobs:
            if j.backend == "newton" and j.status == "pending":
                j.status = "failed"
                j.failure = FailureInfo(
                    kind="newton_floor",
                    message=(
                        f"no host meets Newton CUDA floor "
                        f"{_NEWTON_CUDA_FLOOR[0]}.{_NEWTON_CUDA_FLOOR[1]}; "
                        f"run `odin-cuda install --target "
                        f"{_NEWTON_CUDA_FLOOR[0]}.{_NEWTON_CUDA_FLOOR[1]}`"
                    ),
                    details={"newton_cuda_floor": list(_NEWTON_CUDA_FLOOR)},
                )
                j.ended_at = _utc_now_iso()

    # Seed the state and spawn workers.
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id=dispatch_id,
        started_at=started_at,
        ended_at=None,
        seeds=options.seeds,
        commit_sha=commit_sha,
        fleet=[
            FleetSnapshot(
                host=h.host,
                status="down" if h.host in down_hosts else "idle",
                last_error=None,
            )
            for h in fleet.hosts
        ],
        jobs=merged_jobs,
        skipped=merged_skipped,
    )
    write_dispatch_state(dispatch_dir, state)

    # Enqueue pending jobs.
    job_q: queue.Queue = queue.Queue()
    state_chan: queue.Queue = queue.Queue()
    for j in state.jobs:
        if j.status == "pending":
            job_q.put(j)

    # Snapshot remaining-work count BEFORE workers start. Legacy-PTY workers
    # mutate ``j.status`` directly inside ``_execute``, so once they're alive
    # the counter races against in-flight terminal mutations. Detached workers
    # don't mutate status, but the snapshot is just as correct for them.
    jobs_by_id: dict[str, JobEntry] = {j.run_id: j for j in state.jobs}
    # Both 'pending' jobs (queued for submit) and 'running' jobs (reattached
    # via reconcile in detached mode) count toward "work outstanding".
    remaining = sum(1 for j in state.jobs if j.status in ("pending", "running"))

    # In detached mode, reconcile may have left some jobs in 'running' to
    # signal "trainer is still alive on the remote — keep polling on
    # resume". Seed per-host worker inflight maps before the workers
    # start so they pick up the polling loop without re-submitting.
    reattached_by_host: dict[str, list[JobEntry]] = {}
    if options.detached_mode:
        for j in state.jobs:
            if j.status == "running" and j.assigned_to is not None:
                reattached_by_host.setdefault(j.assigned_to, []).append(j)

    shutdown_event = threading.Event()
    workers: list[ValkyrieWorker] = []
    for host in healthy:
        w = ValkyrieWorker(
            host=host,
            job_queue=job_q,
            state_chan=state_chan,
            dispatch_dir=dispatch_dir,
            options=WorkerOptions(
                per_job_timeout_s=options.per_job_timeout_s,
                max_infrastructure_retries=options.max_infrastructure_retries,
                consecutive_failure_quarantine=options.consecutive_failure_quarantine,
                detached_mode=options.detached_mode,
                poll_interval_s=options.poll_interval_s,
            ),
            ssh=ssh,
            rsync=rsync,
            shutdown_event=shutdown_event,
        )
        for reattach_job in reattached_by_host.get(host.host, []):
            from tools.odin.asgard.worker import JobInflight

            w._inflight[reattach_job.run_id] = JobInflight(
                job=reattach_job,
                tracker=None,
                submitted_at_monotonic=time.monotonic(),
            )
        w.start()
        workers.append(w)

    # Drain state events into state.jobs; rewrite dispatch.json after each.
    retry_db = RetryDB(dispatch_dir.parent)
    live_retry_run_ids: set[str] = set()
    live_retry_poll_s = max(0.05, options.live_retry_poll_s)
    last_retry_poll = time.monotonic()
    last_write = time.monotonic()
    while remaining > 0 and any(w.is_alive() for w in workers):
        try:
            ev: StateEvent = state_chan.get(timeout=min(1.0, live_retry_poll_s))
        except queue.Empty:
            now = time.monotonic()
            if now - last_retry_poll >= live_retry_poll_s:
                added = _consume_live_retries(
                    retry_db=retry_db,
                    dispatch_id=dispatch_id,
                    jobs_by_id=jobs_by_id,
                    job_q=job_q,
                    live_retry_run_ids=live_retry_run_ids,
                )
                if added:
                    remaining += added
                    write_dispatch_state(dispatch_dir, state)
                    last_write = now
                last_retry_poll = now
            if time.monotonic() - last_write >= 5.0:
                write_dispatch_state(dispatch_dir, state)
                last_write = time.monotonic()
            continue
        remaining -= _apply_state_event(state, ev, jobs_by_id, options.verbose)
        _mark_live_retry_consumed(
            retry_db=retry_db,
            dispatch_id=dispatch_id,
            ev=ev,
            live_retry_run_ids=live_retry_run_ids,
        )
        write_dispatch_state(dispatch_dir, state)
        last_write = time.monotonic()
        if ev.transition in {"completed", "failed"}:
            added = _consume_live_retries(
                retry_db=retry_db,
                dispatch_id=dispatch_id,
                jobs_by_id=jobs_by_id,
                job_q=job_q,
                live_retry_run_ids=live_retry_run_ids,
            )
            if added:
                remaining += added
                write_dispatch_state(dispatch_dir, state)
                last_write = time.monotonic()
            last_retry_poll = time.monotonic()

    shutdown_event.set()
    for _ in workers:
        job_q.put(None)
    for w in workers:
        w.join(timeout=30.0)

    _sweep_pending_after_dispatch(state)

    state.ended_at = _utc_now_iso()
    write_dispatch_state(dispatch_dir, state)

    completed_n = sum(1 for j in state.jobs if j.status == "completed")
    failed_n = sum(1 for j in state.jobs if j.status == "failed")
    pending_n = sum(1 for j in state.jobs if j.status == "pending")
    skipped_n = len(state.skipped)
    skip_kinds = ", ".join(sorted({s.reason for s in state.skipped})) or "-"
    print(
        f"odin-dispatch: {completed_n} completed, {failed_n} failed, "
        f"{skipped_n} skipped ({skip_kinds}), {pending_n} pending out of "
        f"{len(state.jobs) + skipped_n} total"
    )

    if not options.skip_aggregate:
        try:
            from tools.odin.valhalla import aggregate_dispatch, write_aggregate

            agg = aggregate_dispatch(dispatch_dir)
            write_aggregate(dispatch_dir, agg)
        except Exception as exc:  # noqa: BLE001 — aggregate failure must not mask the dispatch return
            print(f"[WARNING] aggregate step failed: {exc}")

    return state
