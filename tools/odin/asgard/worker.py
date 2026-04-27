# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ValkyrieWorker — per-host thread that consumes jobs and runs them end-to-end."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.transport import RsyncRunner, SSHResult, SSHRunner

__all__ = ["StateEvent", "ValkyrieWorker", "WorkerOptions"]


@dataclass
class WorkerOptions:
    per_job_timeout_s: int = 14400
    max_infrastructure_retries: int = 2


@dataclass
class StateEvent:
    """Message posted by a worker to the state channel on every transition."""

    run_id: str
    host: str
    transition: str  # "running" | "completed" | "failed" | "shutdown_idle"
    failure: FailureInfo | None = None
    started_at: str | None = None
    ended_at: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# docker exec exits 125 when the docker command itself failed (container not
# found, daemon unreachable). Treat those as infrastructure, not hugin_crash.
_INFRASTRUCTURE_DOCKER_EXIT_CODES = {125, 126, 127}


def _build_docker_exec_cmd(host: ValkyrieConfig, job: JobEntry) -> str:
    """Return the remote shell command to run Hugin/Munin inside the container.

    Shape: ``cd {isaaclab_path} && docker exec {container_name} bash -lc '...'``
    where the inner command is the Hugin (rsl_rl) or Munin (skrl) wrapper
    invocation with the job's CLI args.
    """
    runner_script = "tools/odin/hugin/run.py" if job.framework == "rsl_rl" else "tools/odin/munin/run.py"
    inner_parts = [
        "cd /workspace/isaaclab",
        "PYTHONPATH=.",
        f"./isaaclab.sh -p {runner_script}",
        f"--task {job.task_id}",
        f"--backend {job.backend}",
        f"--seed {job.seed}",
        f"--num_envs {job.num_envs}",
        f"--max_iterations {job.max_iterations}",
        "--runs_root odin_runs",
        f"--run_id {job.run_id}",
    ]
    inner = " && ".join(inner_parts[:1]) + " && " + " ".join(inner_parts[1:])
    return f"cd {host.isaaclab_path} && docker exec {host.container_name} bash -lc '{inner}'"


class ValkyrieWorker(threading.Thread):
    """Per-Valkyrie worker thread.

    Pulls :class:`JobEntry` items from a shared ``queue.Queue`` and runs
    them end-to-end: docker-exec-the-job over SSH, tee stdout to a local
    log file, rsync-pull the bundle back, validate, classify failures.

    Events posted to ``state_chan`` on every transition are consumed by
    the main thread to rewrite ``dispatch.json``.
    """

    def __init__(
        self,
        host: ValkyrieConfig,
        job_queue: queue.Queue,
        state_chan: queue.Queue,
        dispatch_dir: Path,
        options: WorkerOptions,
        *,
        ssh: SSHRunner,
        rsync: RsyncRunner,
        shutdown_event: threading.Event,
    ):
        super().__init__(name=f"ValkyrieWorker-{host.host}", daemon=True)
        self.host = host
        self._job_queue = job_queue
        self._state_chan = state_chan
        self._dispatch_dir = dispatch_dir
        self._options = options
        self._ssh = ssh
        self._rsync = rsync
        self._shutdown = shutdown_event
        self._preferred_not_seen: dict[str, int] = {}

    # -- public entry point -------------------------------------------------

    def run(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:  # sentinel: queue drained
                return
            if job.preferred_not and self.host.host in job.preferred_not:
                # Put it back and let another worker pick it up. To avoid
                # spinning in the degenerate "only worker alive" case, bound
                # the number of times WE will refuse the same job.
                seen_count = self._preferred_not_seen.get(job.run_id, 0) + 1
                self._preferred_not_seen[job.run_id] = seen_count
                if seen_count < 3:
                    self._job_queue.put(job)
                    time.sleep(0.5)
                    continue
                # Fall through: take the job anyway.
            self._execute(job)

    # -- execute one job ----------------------------------------------------

    def _execute(self, job: JobEntry) -> None:
        """Run one job end-to-end with infrastructure retries on this host.

        Infrastructure failures (docker daemon errors, exit codes 125/126/127)
        are retried up to ``max_infrastructure_retries`` times on the *same*
        host before emitting a terminal ``failed`` event.  Cross-host routing
        via ``preferred_not`` is handled by Task 9's bounded fallback.
        """
        # Infrastructure retry loop — only loops on infrastructure failures.
        while True:
            started_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="running", started_at=started_at)
            )
            job.started_at = started_at
            job.attempts += 1

            ssh_tail = self._dispatch_dir / job.bundle_dir_name / "logs" / "ssh-tail.log"
            cmd = _build_docker_exec_cmd(self.host, job)
            ssh_result = self._ssh.run(
                self.host, cmd, timeout_s=float(self._options.per_job_timeout_s), stdout_tee=ssh_tail
            )

            # After an SSH timeout, the local ssh process is terminated, but
            # the ``docker exec``'d training process inside the container
            # often survives (docker's signal-forwarding does not cover this
            # path reliably on every kernel). The zombie keeps burning GPU
            # and will contend for resources with any subsequent job on the
            # same host. Dispatch a best-effort pkill by run_id pattern —
            # failure here is logged to the buffer but never escalated,
            # because the job is already flagged timed-out.
            if ssh_result.timed_out:
                self._cleanup_remote_process(job)

            failure = self._classify(ssh_result, job, ssh_tail)
            if failure is not None and failure.kind == "infrastructure":
                if job.attempts <= self._options.max_infrastructure_retries:
                    # Stay in the retry loop: try again on this host.
                    continue
                # Exhausted retries → emit terminal failure below.

            break  # non-infrastructure result or retries exhausted

        if failure is not None:
            job.status = "failed"
            job.failure = failure
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(
                    run_id=job.run_id,
                    host=self.host.host,
                    transition="failed",
                    failure=failure,
                    ended_at=job.ended_at,
                )
            )
            return

        # Success path: rsync pull the bundle back.
        remote_bundle = f"{self.host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        local_bundle = self._dispatch_dir / job.bundle_dir_name
        rsync_result = self._rsync.pull(self.host, remote_bundle, local_bundle)
        if rsync_result.exit_code != 0:
            job.status = "failed"
            job.failure = FailureInfo(
                kind="infrastructure",
                message=f"rsync pull failed: {rsync_result.stderr.strip() or 'non-zero exit'}",
                details={"attempts": job.attempts},
            )
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=job.failure)
            )
            return

        # Validate the bundle: manifest.json present, schema-v1 shape.
        bundle_failure = _validate_bundle(local_bundle)
        if bundle_failure is not None:
            job.status = "failed"
            job.failure = bundle_failure
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=bundle_failure)
            )
            return

        job.status = "completed"
        job.ended_at = _utc_now_iso()
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="completed",
                ended_at=job.ended_at,
            )
        )

    # -- timeout cleanup ----------------------------------------------------

    def _cleanup_remote_process(self, job: JobEntry) -> None:
        """Best-effort ``pkill -9 -f <run_id>`` inside the Valkyrie's container.

        Run after an SSH timeout to stop the zombie ``docker exec``-launched
        training process from holding the GPU on this host. Uses the job's
        run_id as the pattern — that id is already baked into every
        Hugin/Munin invocation's argv (via ``--run_id``), so the match is
        surgical and will not hit unrelated processes. Errors are
        intentionally swallowed; this is a hygiene step for the *next* job,
        not a prerequisite for reporting the current one as timed out.
        """
        cleanup_cmd = f"docker exec {self.host.container_name} pkill -9 -f '{job.run_id}' 2>/dev/null; true"
        self._ssh.run(self.host, cleanup_cmd, timeout_s=30.0)

    # -- classification -----------------------------------------------------

    def _classify(self, r: SSHResult, job: JobEntry, ssh_tail: Path) -> FailureInfo | None:
        if r.timed_out:
            return FailureInfo(
                kind="timeout",
                message=f"remote process exceeded {self._options.per_job_timeout_s}s",
                details={
                    "duration_s": r.duration_s,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        if r.exit_code in _INFRASTRUCTURE_DOCKER_EXIT_CODES:
            return FailureInfo(
                kind="infrastructure",
                message=(f"docker exec failed with exit {r.exit_code}: {r.stderr.strip() or 'unknown'}"),
                details={
                    "exit_code": r.exit_code,
                    "attempts": job.attempts,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        if r.exit_code != 0:
            stderr_text = r.stderr or ""
            if "preset_unsupported:" in stderr_text:
                return FailureInfo(
                    kind="preset_unsupported",
                    message="benchmark script reported missing preset",
                    details={
                        "exit_code": r.exit_code,
                        "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                    },
                )
            _last_line = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else None
            _stderr_tail = repr(_last_line) if _last_line is not None else "(empty)"
            return FailureInfo(
                kind="hugin_crash",
                message=f"exit code {r.exit_code}; stderr tail: {_stderr_tail}",
                details={
                    "exit_code": r.exit_code,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        return None


def _validate_bundle(local_bundle: Path) -> FailureInfo | None:
    """Check that manifest.json exists and declares schema_version==1.0."""
    manifest_path = local_bundle / "manifest.json"
    if not manifest_path.exists():
        return FailureInfo(
            kind="hugin_malformed_bundle",
            message="manifest.json missing after rsync pull",
            details={"bundle_dir": str(local_bundle.name)},
        )
    try:
        m = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return FailureInfo(
            kind="hugin_malformed_bundle",
            message=f"manifest.json is not valid JSON: {exc}",
            details={"bundle_dir": str(local_bundle.name)},
        )
    if str(m.get("schema_version", "")) != "1.0":
        return FailureInfo(
            kind="hugin_malformed_bundle",
            message=f"manifest.json schema_version != 1.0 (got {m.get('schema_version')!r})",
            details={"bundle_dir": str(local_bundle.name)},
        )
    return None
