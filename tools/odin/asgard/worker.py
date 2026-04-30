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
from tools.odin.asgard.recovery import recover_valkyrie_gpu
from tools.odin.asgard.transport import RsyncRunner, SSHResult, SSHRunner

__all__ = ["StateEvent", "ValkyrieWorker", "WorkerOptions"]


@dataclass
class _ConsecutiveFailureTracker:
    """Per-worker counter for consecutive job failures.

    Tracks how many failures have happened in a row on this worker without
    an intervening completed job. When the threshold is reached, the
    worker should quarantine its host (emit ``host_down``) and exit.

    A ``threshold`` of ``0`` disables the circuit-breaker entirely.

    Args:
        threshold: Number of consecutive failures that triggers quarantine.
            ``0`` disables the breaker.
    """

    threshold: int
    count: int = 0

    def note_failure(self) -> bool:
        """Record a failure.

        Returns:
            ``True`` iff the threshold has been reached (caller should
            quarantine). ``False`` when the breaker is disabled or the
            count is still below threshold.
        """
        if self.threshold <= 0:
            return False
        self.count += 1
        return self.count >= self.threshold

    def note_success(self) -> None:
        """Reset the counter to zero after a completed job."""
        self.count = 0


@dataclass
class WorkerOptions:
    per_job_timeout_s: int = 43200
    max_infrastructure_retries: int = 2
    consecutive_failure_quarantine: int = 3  # 0 = disabled


@dataclass
class StateEvent:
    """Message posted by a worker to the state channel on every transition.

    ``transition`` values:

    - ``running``: job dispatched to host; ``started_at`` set.
    - ``completed``: job finished, bundle pulled, manifest validated.
    - ``failed``: terminal failure for this job; ``failure`` set.
    - ``recovered``: GPU loss detected, recovery succeeded; retry follows.
    - ``host_down``: GPU loss detected, recovery failed; host transitions
      to ``status="down"``. The worker re-queues the in-flight job (so
      another healthy worker can pick it up) and stops pulling further
      jobs from the queue (its own ``_down_event`` is set).
    - ``shutdown_idle``: worker received its sentinel and exited cleanly.
    """

    run_id: str
    host: str
    transition: str
    failure: FailureInfo | None = None
    started_at: str | None = None
    ended_at: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# docker exec exits 125 when the docker command itself failed (container not
# found, daemon unreachable). Treat those as infrastructure, not hugin_crash.
_INFRASTRUCTURE_DOCKER_EXIT_CODES = {125, 126, 127}

# Failure kinds that count toward the per-host consecutive-failure circuit
# breaker. Job-level failures (timeout, hugin_crash, preset_unsupported)
# don't say anything about host health — a slow training task or a missing
# preset is the same on every host — so they are excluded. Host-health
# failures (stuck container, corrupt bundle pull, post-recovery GPU loss)
# are the ones that suggest "this Valkyrie is wedged; route around it".
_HOST_HEALTH_FAILURE_KINDS = frozenset({"infrastructure", "hugin_malformed_bundle", "gpu_lost"})

# GPU-loss stderr signatures recognised by ``_classify``.  Worker emits
# FailureInfo(kind="gpu_lost") when the training process exited non-zero
# AND its stderr contains any of these strings.  Recovery (T8) is then
# attempted via container restart before retrying on the same host.
#
# ``odin: gpu_unavailable`` is the marker emitted by the pre-job
# ``nvidia-smi -L`` probe in :func:`_build_docker_exec_cmd` — when the
# probe fails, the rest of the SSH/docker pipeline is skipped and we get
# this marker on SSH-side stderr without having to read remote bundle
# logs. Catches the case where a previous job's training left the GPU /
# NVML state wedged.
_GPU_LOST_SIGNATURES = (
    "Failed to initialize NVML",
    "CUDA error: no CUDA-capable device is detected",
    "Vulkan ERROR_INCOMPATIBLE_DRIVER",
    "odin: gpu_unavailable",
)


def _build_docker_exec_cmd(host: ValkyrieConfig, job: JobEntry) -> str:
    """Return the remote shell command to run Hugin/Munin inside the container.

    Calls ``_isaac_sim/python.sh`` directly (not ``./isaaclab.sh -p``) — the
    outer wrapper's ``set -e`` + ``error_exit`` trap discards child stderr
    on non-zero exit, hiding real tracebacks from the bundle.

    Stdout and stderr are redirected into bundle-local log files so they
    survive the rsync-back regardless of exit code.

    Args:
        host: Valkyrie host configuration.
        job: Job metadata used to build the runner invocation.

    Returns:
        Shell command of shape
        ``cd <isaaclab_path> && docker exec <container_name> bash -lc '...'``
        ready to pass to :class:`SSHRunner.run`.
    """
    runner_script = "tools/odin/hugin/run.py" if job.framework == "rsl_rl" else "tools/odin/munin/run.py"
    bundle_logs = f"odin_runs/{job.bundle_dir_name}/logs"
    # Pre-job nvidia-smi probe. If a prior job left the GPU / NVML state
    # wedged, ``nvidia-smi -L`` fails fast and we surface a recognizable
    # marker on SSH-side stderr so :data:`_GPU_LOST_SIGNATURES` matches in
    # :meth:`ValkyrieWorker._classify`. The probe's own stderr (e.g.
    # ``Failed to initialize NVML: Unknown Error``) is appended after the
    # marker for diagnostic context. Without this probe, training scripts
    # crash deep in PyTorch's CUDA init with "No CUDA GPUs are available",
    # which only lands in the bundle's training.stderr.log on the remote
    # — invisible to the worker's classifier.
    gpu_probe = (
        f"(nvidia-smi -L >/dev/null 2>{bundle_logs}/nvidia-probe.log || "
        f"{{ echo \"odin: gpu_unavailable: $(tr -d '\\n' < {bundle_logs}/nvidia-probe.log)\" >&2; exit 1; }})"
    )
    inner = (
        f"cd /workspace/isaaclab "
        f"&& mkdir -p {bundle_logs} "
        f"&& {gpu_probe} "
        f"&& PYTHONPATH=. _isaac_sim/python.sh {runner_script}"
        f" --task {job.task_id}"
        f" --backend {job.backend}"
        f" --seed {job.seed}"
        f" --num_envs {job.num_envs}"
        f" --max_iterations {job.max_iterations}"
        f" --runs_root odin_runs"
        f" --run_id {job.run_id}"
        f" > {bundle_logs}/hugin-stdout.log"
        f" 2> {bundle_logs}/hugin-stderr.log"
    )
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
        # Set when this worker's host is marked down (gpu_lost recovery
        # failed). Causes ``run()`` to stop pulling new jobs.
        self._down_event = threading.Event()
        self._preferred_not_seen: dict[str, int] = {}
        self._fail_tracker = _ConsecutiveFailureTracker(threshold=options.consecutive_failure_quarantine)

    # -- public entry point -------------------------------------------------

    def run(self) -> None:
        while not self._shutdown.is_set() and not self._down_event.is_set():
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
            elif failure is not None and failure.kind == "gpu_lost":
                rec = recover_valkyrie_gpu(self.host, ssh=self._ssh)
                if rec.recovered:
                    self._state_chan.put(
                        StateEvent(
                            run_id=job.run_id,
                            host=self.host.host,
                            transition="recovered",
                        )
                    )
                    if job.attempts <= self._options.max_infrastructure_retries:
                        continue
                    # Retries exhausted on this host even after successful recovery.
                else:
                    # Recovery failed: this host is down. Re-queue the job so
                    # another healthy worker can pick it up via the existing
                    # bounded-fallback (preferred_not) routing, mark this host
                    # in preferred_not, and stop pulling further jobs from
                    # this worker. The runner sweeps any still-pending jobs
                    # at the end of the dispatch when no host can run them.
                    self._state_chan.put(
                        StateEvent(
                            run_id=job.run_id,
                            host=self.host.host,
                            transition="host_down",
                            failure=failure,
                        )
                    )
                    job.preferred_not = set(job.preferred_not) | {self.host.host}
                    self._job_queue.put(job)
                    self._down_event.set()
                    return  # Do not terminal-fail; job is back on the queue.

            break  # non-recoverable result or retries exhausted

        if failure is not None:
            if self._quarantine_check_and_handle(job=job, failure=failure, ssh_tail=ssh_tail):
                return
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
            rsync_failure = FailureInfo(
                kind="infrastructure",
                message=f"rsync pull failed: {rsync_result.stderr.strip() or 'non-zero exit'}",
                details={"attempts": job.attempts},
            )
            if self._quarantine_check_and_handle(job=job, failure=rsync_failure, ssh_tail=ssh_tail):
                return
            job.status = "failed"
            job.failure = rsync_failure
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=rsync_failure)
            )
            return

        # Validate the bundle: manifest.json present, schema-v1 shape.
        bundle_failure = _validate_bundle(local_bundle)
        if bundle_failure is not None:
            if self._quarantine_check_and_handle(job=job, failure=bundle_failure, ssh_tail=ssh_tail):
                return
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
        self._fail_tracker.note_success()

    # -- circuit-breaker helper ---------------------------------------------

    def _quarantine_check_and_handle(self, *, job: JobEntry, failure: FailureInfo, ssh_tail: Path) -> bool:
        """Increment the failure tracker and, if quarantine triggers, emit
        host_down + re-queue the job + arm ``_down_event``.

        Only host-health failures (see :data:`_HOST_HEALTH_FAILURE_KINDS`)
        count toward the breaker; job-level failures (timeout, hugin_crash,
        preset_unsupported) do not. This avoids cascading a slow training
        task or a misconfigured preset into a host quarantine.

        If the threshold isn't reached, this method does NOT emit any
        event — caller is expected to post the regular ``failed`` event.

        Args:
            job: The job that just failed.
            failure: The :class:`FailureInfo` describing the failure.
            ssh_tail: Path to the ssh-tail log (passed through for context;
                not used directly but available for future diagnostics).

        Returns:
            ``True`` iff the worker should exit (threshold reached).
            ``False`` if the worker should continue with the next job.
        """
        if failure.kind not in _HOST_HEALTH_FAILURE_KINDS:
            return False
        if not self._fail_tracker.note_failure():
            return False
        # Threshold reached: re-queue the triggering job so another healthy
        # host can pick it up, and quarantine this host.
        job.preferred_not = set(job.preferred_not) | {self.host.host}
        self._job_queue.put(job)
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="host_down",
                failure=FailureInfo(
                    kind="circuit_breaker",
                    message=(
                        f"{self._fail_tracker.threshold} consecutive failures on {self.host.host}; quarantining host"
                    ),
                ),
            )
        )
        self._down_event.set()
        return True

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
            if any(sig in stderr_text for sig in _GPU_LOST_SIGNATURES):
                return FailureInfo(
                    kind="gpu_lost",
                    message="GPU-loss signature in stderr",
                    details={
                        "exit_code": r.exit_code,
                        "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                    },
                )
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
