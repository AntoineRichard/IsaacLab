# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resume reconciliation — adopt or kill orphan jobs left on remotes.

Called once on ``--resume`` before workers spin up. For each job that the
prior dispatch state had in ``running`` status (and thus has an
``assigned_to`` host), check the remote bundle directory and process
state, and pick one of six outcomes:

1. ``adopted_completed`` — remote ``manifest.json`` shows both phases
   ``completed`` / ``exit_code: 0``. Rsync the bundle back; mark
   ``completed``.
2. ``adopted_failed`` — remote ``manifest.json`` exists but a phase
   failed. Rsync the bundle back; mark ``failed`` with a Hugin-crash
   classification. (Detached mode also lands here when the trainer
   exited without writing a manifest — see ``classify_remote_stderr``.)
3. ``reattached_inflight`` — detached mode only: no manifest yet but the
   poll script reports the trainer is still alive. Leave the job in
   ``running`` so the worker can keep polling it after resume.
4. ``killed_alive_orphan`` — legacy mode: no manifest, but ``pgrep``
   finds a process matching ``--run_id <run_id>``. Kill it (``pkill
   -9``); mark ``pending`` for re-dispatch. (Not used in detached mode
   — the trainer is intentionally outliving the dispatcher there, and
   we re-attach instead of killing.)
5. ``dead_re_pending`` — no manifest, no live process. Mark
   ``pending`` for re-dispatch.
6. (skipped) — ``assigned_to`` is None, nothing to reconcile.

The caller's existing ``reset_in_flight_to_pending`` then handles any
remaining ``running`` rows (which would only be cases 4 / 5 / 6 above
that we've already mutated to ``pending``, or unrelated edge cases).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.transport import RsyncRunner, SSHRunner

__all__ = ["ReconcileOutcome", "reconcile_orphans"]


_REMOTE_RUNS_ROOT = "/workspace/isaaclab/odin_runs"


class _CancelDBLike(Protocol):
    """Structural type for the bits of CancelDB this module uses.

    Avoids the dashboard → asgard import cycle that a direct
    :class:`~tools.odin.valhalla.dashboard.cancel_db.CancelDB` import
    would create, while still giving callers static-analysis coverage.
    """

    def read_pending(self, dispatch_id: str) -> dict[str, str]: ...

    def mark_consumed(self, dispatch_id: str, run_id: str, *, outcome: str) -> None: ...


@dataclass(frozen=True)
class ReconcileOutcome:
    run_id: str
    # One of: adopted_completed, adopted_failed, reattached_inflight,
    # killed_alive_orphan, dead_re_pending.
    action: str


def _host_by_name(fleet: Fleet, name: str | None) -> ValkyrieConfig | None:
    if name is None:
        return None
    for h in fleet.hosts:
        if h.host == name:
            return h
    return None


def _read_remote_manifest(host: ValkyrieConfig, run_id: str, ssh: SSHRunner) -> dict | None:
    """Cat the remote manifest.json. Return parsed dict or None if absent/invalid."""
    cmd = f"docker exec {host.container_name} cat {_REMOTE_RUNS_ROOT}/{run_id}/manifest.json"
    r = ssh.run(host, cmd, timeout_s=15.0)
    if r.exit_code != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _manifest_indicates_clean_completion(manifest: dict) -> bool:
    phases = manifest.get("phases", {})
    if not phases:
        return False
    return all(phase.get("status") == "completed" and phase.get("exit_code") == 0 for phase in phases.values())


def _process_alive(host: ValkyrieConfig, run_id: str, ssh: SSHRunner) -> bool:
    cmd = f"docker exec {host.container_name} pgrep -f -- '--run_id {run_id}'"
    r = ssh.run(host, cmd, timeout_s=10.0)
    return r.exit_code == 0 and bool(r.stdout.strip())


def _kill_remote(host: ValkyrieConfig, run_id: str, ssh: SSHRunner) -> None:
    cmd = f"docker exec {host.container_name} pkill -9 -f -- '--run_id {run_id}' 2>/dev/null; true"
    ssh.run(host, cmd, timeout_s=10.0)


def _pull_bundle(host: ValkyrieConfig, run_id: str, dispatch_dir: Path, rsync: RsyncRunner) -> bool:
    remote_path = f"{host.isaaclab_path}/odin_runs/{run_id}"
    local_path = dispatch_dir / run_id
    rr = rsync.pull(host, remote_path, local_path)
    return rr.exit_code == 0


def _detached_poll_one(host: ValkyrieConfig, bundle_id: str, ssh: SSHRunner) -> str | None:
    """Run the detached-mode poll script for a single bundle, return its state.

    Returns ``None`` when the SSH call itself failed — the caller should
    treat that as "no information; fall through to legacy handling".
    """
    from tools.odin.asgard.worker import _build_poll_script, _parse_poll_output

    cmd = _build_poll_script(host, [bundle_id])
    r = ssh.run(host, cmd, timeout_s=30.0, pty=False)
    if r.exit_code != 0:
        return None
    return _parse_poll_output(r.stdout).get(bundle_id)


def _classify_pulled_bundle(local_bundle: Path) -> FailureInfo:
    """Classify a failed run from the locally-pulled bundle's stderr files.

    Reads ``logs/odin-submit-error.log`` and ``logs/hugin-stderr.log``
    and runs the combined text through
    :func:`~tools.odin.asgard.worker.classify_remote_stderr` so reconcile
    and worker reach the same kind for the same input.
    """
    from tools.odin.asgard.worker import classify_remote_stderr

    submit_err = local_bundle / "logs" / "odin-submit-error.log"
    train_err = local_bundle / "logs" / "hugin-stderr.log"
    parts: list[str] = []
    if submit_err.exists():
        parts.append(submit_err.read_text())
    if train_err.exists():
        parts.append(train_err.read_text())
    failure = classify_remote_stderr("\n".join(parts))
    failure.details = {**failure.details, "reconciled": True}
    return failure


def reconcile_orphans(
    *,
    fleet: Fleet,
    jobs: list[JobEntry],
    dispatch_dir: Path,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    detached_mode: bool = False,
    cancel_db: _CancelDBLike | None = None,
) -> list[ReconcileOutcome]:
    """Reconcile every ``running`` job against its prior remote host.

    Mutates ``jobs`` in place (status, failure, assigned_to). The caller
    must persist the mutated state via :func:`write_dispatch_state`.

    Detached-mode behaviour adds two new outcomes:

      - ``reattached_inflight``: the poll script reports the trainer is
        still alive on the remote. Leave the job in ``running`` so the
        worker can pick the polling back up; do NOT pull the bundle yet
        (training is mid-write).
      - ``adopted_failed`` (extended): the poll reports
        ``exited-no-manifest``. Pull the bundle (best effort) and
        classify via remote stderr, mirroring the worker's terminal-fail
        path in :meth:`ValkyrieWorker._finalize_terminal`.

    The legacy-mode behaviour is unchanged: ``pgrep`` + ``pkill`` on no
    manifest, mark ``pending``.

    Args:
        fleet: Fleet config used to resolve ``assigned_to`` → host.
        jobs: All jobs from the prior :class:`DispatchState`. Only those
            in ``running`` status with a non-None ``assigned_to`` are
            reconciled; others are skipped.
        dispatch_dir: Local dispatch directory; bundles are rsynced here.
        ssh: SSH runner.
        rsync: Rsync runner.
        detached_mode: When ``True``, use the detached-mode poll script
            instead of ``pgrep`` for the no-manifest fall-through, and
            re-attach in-flight jobs. Default ``False`` preserves the
            legacy behaviour (and matches existing call sites).
        cancel_db: Optional :class:`CancelDB` (passed by the runner on
            ``--resume``). When non-None, pending skip/kill rows are
            re-applied before workers spin up — skips flip pending jobs
            to failed; kills are applied to in-flight jobs after the
            re-attach by seeding ``worker._cancel_request`` (handled in
            the runner, not here).

    Returns:
        List of :class:`ReconcileOutcome` — one per reconciled job.
        Skipped jobs do not appear in the output.
    """
    outcomes: list[ReconcileOutcome] = []
    for j in jobs:
        if j.status != "running":
            continue
        host = _host_by_name(fleet, j.assigned_to)
        if host is None:
            continue

        manifest = _read_remote_manifest(host, j.run_id, ssh)
        if manifest is not None:
            if _manifest_indicates_clean_completion(manifest):
                _pull_bundle(host, j.run_id, dispatch_dir, rsync)
                j.status = "completed"
                outcomes.append(ReconcileOutcome(run_id=j.run_id, action="adopted_completed"))
            else:
                _pull_bundle(host, j.run_id, dispatch_dir, rsync)
                j.status = "failed"
                j.failure = FailureInfo(
                    kind="hugin_crash",
                    message="adopted from remote manifest after orphan reconciliation",
                    details={"reconciled": True},
                )
                outcomes.append(ReconcileOutcome(run_id=j.run_id, action="adopted_failed"))
            continue

        if detached_mode:
            poll_state = _detached_poll_one(host, j.bundle_dir_name, ssh)
            if poll_state == "done":
                # Manifest landed between the first check and the poll
                # (or the upstream manifest read returned None spuriously).
                # Pull and adopt as completed.
                _pull_bundle(host, j.run_id, dispatch_dir, rsync)
                j.status = "completed"
                outcomes.append(ReconcileOutcome(run_id=j.run_id, action="adopted_completed"))
                continue
            if poll_state == "alive":
                # Trainer is still running — leave 'running', let the
                # worker re-attach and keep polling.
                outcomes.append(ReconcileOutcome(run_id=j.run_id, action="reattached_inflight"))
                continue
            if poll_state == "exited-no-manifest":
                _pull_bundle(host, j.run_id, dispatch_dir, rsync)
                j.status = "failed"
                j.failure = _classify_pulled_bundle(dispatch_dir / j.bundle_dir_name)
                outcomes.append(ReconcileOutcome(run_id=j.run_id, action="adopted_failed"))
                continue
            if poll_state == "no-pidfile":
                # Submit was interrupted before the pidfile write — no
                # trainer ever ran. Re-pending.
                j.status = "pending"
                j.assigned_to = None
                j.started_at = None
                outcomes.append(ReconcileOutcome(run_id=j.run_id, action="dead_re_pending"))
                continue
            # poll_state is None (SSH failed) → fall through to legacy path
            # so we still get a sane outcome for the operator.

        if _process_alive(host, j.run_id, ssh):
            _kill_remote(host, j.run_id, ssh)
            j.status = "pending"
            j.assigned_to = None
            j.started_at = None
            outcomes.append(ReconcileOutcome(run_id=j.run_id, action="killed_alive_orphan"))
        else:
            j.status = "pending"
            j.assigned_to = None
            j.started_at = None
            outcomes.append(ReconcileOutcome(run_id=j.run_id, action="dead_re_pending"))

    if cancel_db is not None:
        dispatch_id = dispatch_dir.name
        for run_id, kind in list(cancel_db.read_pending(dispatch_id).items()):
            job = next((j for j in jobs if j.run_id == run_id), None)
            if job is None or job.status in {"completed", "failed"}:
                cancel_db.mark_consumed(dispatch_id, run_id, outcome="noop")
                continue
            if kind == "skip" and job.status == "pending":
                job.status = "failed"
                job.failure = FailureInfo(
                    kind="skipped",
                    message="operator skipped before dispatch (applied at resume)",
                    details={"reconciled": True},
                )
                cancel_db.mark_consumed(dispatch_id, run_id, outcome="skipped")
                outcomes.append(ReconcileOutcome(run_id=run_id, action="adopted_failed"))
            # Skip-on-running and pending kill rows are intentionally left
            # untouched here. The runner's _consume_cancellations sees them
            # on its first tick: skip-on-running gets upgraded to kill via
            # CancelDB.upgrade_to_kill, and kill rows fire request_cancel on
            # the assigned worker.

    return outcomes
