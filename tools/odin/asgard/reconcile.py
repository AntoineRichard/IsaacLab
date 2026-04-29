# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resume reconciliation — adopt or kill orphan jobs left on remotes.

Called once on ``--resume`` before workers spin up. For each job that the
prior dispatch state had in ``running`` status (and thus has an
``assigned_to`` host), check the remote bundle directory and process
state, and pick one of five outcomes:

1. ``adopted_completed`` — remote ``manifest.json`` shows both phases
   ``completed`` / ``exit_code: 0``. Rsync the bundle back; mark
   ``completed``.
2. ``adopted_failed`` — remote ``manifest.json`` exists but a phase
   failed. Rsync the bundle back; mark ``failed`` with a Hugin-crash
   classification.
3. ``killed_alive_orphan`` — no manifest yet but a process matching
   ``--run_id <run_id>`` is alive. Kill it (``pkill -9``); mark
   ``pending`` for re-dispatch.
4. ``dead_re_pending`` — no manifest, no live process. Mark
   ``pending`` for re-dispatch.
5. (skipped) — ``assigned_to`` is None, nothing to reconcile.

The caller's existing ``reset_in_flight_to_pending`` then handles any
remaining ``running`` rows (which would only be cases 3 / 4 / 5 above
that we've already mutated to ``pending``, or unrelated edge cases).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.transport import RsyncRunner, SSHRunner

__all__ = ["ReconcileOutcome", "reconcile_orphans"]


_REMOTE_RUNS_ROOT = "/workspace/isaaclab/odin_runs"


@dataclass(frozen=True)
class ReconcileOutcome:
    run_id: str
    action: str  # one of: adopted_completed, adopted_failed, killed_alive_orphan, dead_re_pending


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


def reconcile_orphans(
    *,
    fleet: Fleet,
    jobs: list[JobEntry],
    dispatch_dir: Path,
    ssh: SSHRunner,
    rsync: RsyncRunner,
) -> list[ReconcileOutcome]:
    """Reconcile every ``running`` job against its prior remote host.

    Mutates ``jobs`` in place (status, failure, assigned_to). The caller
    must persist the mutated state via :func:`write_dispatch_state`.

    Args:
        fleet: Fleet config used to resolve ``assigned_to`` → host.
        jobs: All jobs from the prior :class:`DispatchState`. Only those
            in ``running`` status with a non-None ``assigned_to`` are
            reconciled; others are skipped.
        dispatch_dir: Local dispatch directory; bundles are rsynced here.
        ssh: SSH runner.
        rsync: Rsync runner.

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

    return outcomes
