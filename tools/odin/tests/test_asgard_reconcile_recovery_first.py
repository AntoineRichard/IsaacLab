# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression — :func:`reconcile_orphans` attempts host recovery before flipping to pending.

The 2026-05-05 incident saw three Shadow-Vision rows on a transiently
unreachable host get flipped to ``pending`` at dispatcher restart. The
desired behaviour is to attempt one recovery first; on success, leave the
job in ``running`` so Heimdall (or the next dispatch loop) can catch up.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.reconcile import reconcile_orphans
from tools.odin.asgard.recovery import RecoveryResult
from tools.odin.asgard.transport import RsyncResult, SSHResult


@dataclass
class _UnreachableSSH:
    """SSH runner where every probe fails with exit 255 (simulates orphan host)."""

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        return SSHResult(exit_code=255, stdout="", stderr="ssh: connect failed", duration_s=0.5)


@dataclass
class _NoopRsync:
    def pull(self, host, remote_path, local_path):
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def push(self, host, local_path, remote_path):
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _running_job(run_id: str, host: str) -> JobEntry:
    j = JobEntry(
        run_id=run_id,
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=0,
        bundle_dir_name=run_id,
    )
    j.transition_to("running", assigned_to=host, now="2026-05-08T14:00:00Z")
    return j


def _fleet_with_a() -> Fleet:
    return Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])


def test_reconcile_orphan_attempts_recovery_before_flip_to_pending(tmp_path):
    """Recovery succeeds → job stays running."""
    job = _running_job("run-1", "host-a")
    recovery_calls: list[str] = []

    def recover_fn(host, ssh):
        recovery_calls.append(host.host)
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=2.0,
            message="recovered_via_container_restart",
        )

    reconcile_orphans(
        fleet=_fleet_with_a(),
        jobs=[job],
        dispatch_dir=tmp_path,
        ssh=_UnreachableSSH(),
        rsync=_NoopRsync(),
        cancel_db=None,
        recover_fn=recover_fn,
    )
    assert recovery_calls == ["host-a"]
    assert job.status == "running"


def test_reconcile_orphan_flips_to_pending_when_recovery_fails(tmp_path):
    """Recovery fails → existing flip-to-pending behaviour applies."""
    job = _running_job("run-1", "host-a")

    def recover_fn(host, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=2.0,
            message="docker_restart_failed",
        )

    reconcile_orphans(
        fleet=_fleet_with_a(),
        jobs=[job],
        dispatch_dir=tmp_path,
        ssh=_UnreachableSSH(),
        rsync=_NoopRsync(),
        cancel_db=None,
        recover_fn=recover_fn,
    )
    assert job.status == "pending"
