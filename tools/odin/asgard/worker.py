# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ValkyrieWorker — per-host thread that consumes jobs and runs them end-to-end."""

from __future__ import annotations

import json
import queue
import textwrap
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.recovery import recover_valkyrie_gpu
from tools.odin.asgard.tracker import TRACKER_SCHEMA_VERSION
from tools.odin.asgard.transport import RsyncRunner, SSHResult, SSHRunner

__all__ = [
    "JobInflight",
    "POLL_ALIVE",
    "POLL_DONE",
    "POLL_EXITED_NO_MANIFEST",
    "POLL_NO_PIDFILE",
    "StateEvent",
    "SubmitResult",
    "ValkyrieWorker",
    "WorkerOptions",
]


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
    """Per-worker tunables.

    Args:
        per_job_timeout_s: Wall-clock timeout per training run [s].
        max_infrastructure_retries: Cap on per-job infrastructure retries.
        consecutive_failure_quarantine: Per-host consecutive-failure threshold;
            ``0`` disables the circuit breaker.
        detached_mode: When ``True`` the worker uses the submit-and-poll model
            so SSH disconnects no longer kill in-flight training. ``False``
            preserves the legacy single-PTY-per-job behaviour for rollback.
        poll_interval_s: Sleep [s] between poll ticks in detached mode.
        submit_max_retries: Cap on transient-SSH retries during submit. Set
            small (default 3) so that a 1-second glitch at submit doesn't
            kill a job before it starts, but a genuinely-down host fails
            fast enough to free the worker for the next job.
        submit_retry_backoff_s: Sleep [s] between submit retries.
    """

    per_job_timeout_s: int = 43200
    max_infrastructure_retries: int = 2
    consecutive_failure_quarantine: int = 3  # 0 = disabled
    # Default ``False`` for backward compatibility with the existing legacy
    # PTY tests. The dispatcher's :class:`DispatchOptions.detached_mode`
    # default is ``True`` and overrides this when the worker is built via
    # :func:`run_dispatch`.
    detached_mode: bool = False
    poll_interval_s: float = 30.0
    submit_max_retries: int = 3
    submit_retry_backoff_s: float = 2.0


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


def classify_remote_stderr(text: str) -> FailureInfo:
    """Map a remote-stderr blob → :class:`FailureInfo`.

    Pure function (no I/O). Used by both the worker's detached-mode
    finalize path and the reconcile path so the same signature set
    classifies failures regardless of how the bytes reached the
    dispatcher.

    Args:
        text: Concatenated ``odin-submit-error.log`` plus
            ``hugin-stderr.log`` content from the bundle.

    Returns:
        ``FailureInfo`` with kind ``gpu_lost`` / ``preset_unsupported``
        / ``hugin_crash``.
    """
    if any(sig in text for sig in _GPU_LOST_SIGNATURES):
        return FailureInfo(
            kind="gpu_lost",
            message="GPU-loss signature in remote stderr",
            details={"stderr_tail": text.strip()[-400:]},
        )
    if "preset_unsupported:" in text:
        return FailureInfo(
            kind="preset_unsupported",
            message="benchmark script reported missing preset",
            details={"stderr_tail": text.strip()[-400:]},
        )
    last_line = text.strip().splitlines()[-1] if text.strip() else None
    return FailureInfo(
        kind="hugin_crash",
        message=f"trainer exited without manifest; stderr tail: {repr(last_line) if last_line else '(empty)'}",
        details={"stderr_tail": text.strip()[-400:]},
    )


# --- Detached submit / poll -------------------------------------------------

# Poll-output state strings, in sync with the bash :func:`_build_poll_script`.
POLL_DONE = "done"
POLL_ALIVE = "alive"
POLL_EXITED_NO_MANIFEST = "exited-no-manifest"
POLL_NO_PIDFILE = "no-pidfile"

_POLL_STATES = frozenset({POLL_DONE, POLL_ALIVE, POLL_EXITED_NO_MANIFEST, POLL_NO_PIDFILE})

_SUBMIT_OK_PREFIX = "odin-submit: ok"


@dataclass
class SubmitResult:
    """Outcome of a single ``_submit_job`` call.

    Attributes:
        ok: ``True`` when the submit landed and the trainer is now detached
            on the remote. ``False`` for any synchronous-failure path.
        failure: When ``ok`` is ``False``, the classified
            :class:`~tools.odin.asgard.jobs.FailureInfo` to terminal-fail
            the job with.
    """

    ok: bool
    failure: FailureInfo | None = None


@dataclass
class JobInflight:
    """In-memory record of a detached job between submit and finalize.

    Attributes:
        job: The :class:`JobEntry` we submitted.
        tracker: The :class:`~tools.odin.asgard.tracker.Tracker` object
            describing the remote run. ``None`` when the runtime
            constructed the inflight before reading the remote tracker
            (resume re-attach path).
        submitted_at_monotonic: ``time.monotonic()`` snapshot at submit;
            used by ``_sweep_timeouts`` for budget enforcement. Not the
            tracker's ``submitted_at`` (host clocks may drift).
        timeout_kill_dispatched: ``True`` once ``_sweep_timeouts`` has
            issued a best-effort pkill for this run; the next
            ``exited-no-manifest`` poll classifies as ``timeout`` rather
            than re-running ``_classify_remote``.
        kill_dispatched: ``True`` once :meth:`_sweep_cancellations` has
            issued a best-effort pkill in response to an operator kill;
            the next ``exited-no-manifest`` poll classifies as
            ``killed`` rather than running ``_classify_remote``.
    """

    job: JobEntry
    tracker: object | None = None
    submitted_at_monotonic: float = field(default_factory=time.monotonic)
    timeout_kill_dispatched: bool = False
    kill_dispatched: bool = False


def _build_submit_script(
    host: ValkyrieConfig,
    job: JobEntry,
    *,
    submitted_at: str,
    per_job_timeout_s: int,
) -> str:
    """Return the SSH command that detaches the trainer on the remote.

    The command is one outer SSH that pipes a quoted heredoc into
    ``docker exec -i ... bash -l``. The inner script:

      1. ``cd`` into ``/workspace/isaaclab``.
      2. Runs ``nvidia-smi -L``; on failure, writes the
         ``odin: gpu_unavailable`` marker to
         ``<bundle>/logs/odin-submit-error.log`` and exits non-zero so the
         dispatcher's submit phase short-circuits to a ``gpu_lost``
         failure (no need to rsync — the marker is already on remote).
      3. Backgrounds the trainer with ``nohup setsid bash -c '...' &``.
         The inner-inner shell writes its own PID into ``.run.pid``
         (``$$``) and ``exec``s the trainer with stdio redirected to
         bundle-local log files.
      4. Captures ``$!`` (the setsid bash's host-side PID) and writes
         ``.tracker.json`` with all dispatcher-known fields plus the
         captured PID.
      5. Echoes the ``odin-submit: ok`` sentinel and exits 0.

    The outer heredoc is ``<<'ASGARD_SUBMIT_EOF'`` (quoted), so the
    remote login shell does NOT pre-expand ``$$`` / ``$!`` /
    ``$TRAINING_PID`` before sending the body to ``bash -l``. Inside the
    container, all expansion happens in the inner bash, which is the
    only context where those variables have the right values.

    Args:
        host: Target Valkyrie host.
        job: Job to submit. ``job.framework`` selects ``hugin/run.py`` vs
            ``munin/run.py``.
        submitted_at: UTC ISO-8601 timestamp stamped into the tracker for
            audit (not used for timeout enforcement — that uses
            ``time.monotonic()`` on the dispatcher).
        per_job_timeout_s: Tracker-stamped budget for orphan recovery to
            see at resume time.

    Returns:
        Single-string SSH command suitable for
        :meth:`SSHRunner.run` with ``pty=False``.
    """
    runner_script = "tools/odin/hugin/run.py" if job.framework == "rsl_rl" else "tools/odin/munin/run.py"
    bundle = f"odin_runs/{job.bundle_dir_name}"
    bundle_logs = f"{bundle}/logs"
    inner_inner = (
        f"echo $$ > {bundle}/.run.pid; "
        f"exec env PYTHONPATH=. _isaac_sim/python.sh {runner_script}"
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
    body = textwrap.dedent(
        f"""\
        set -u
        cd /workspace/isaaclab
        # Wipe any prior attempt's bundle so a stale manifest / pidfile /
        # tracker doesn't trip up the new run's first poll. A SIGKILL'd or
        # crashed predecessor can leave a manifest.json with status=failed,
        # which the worker would otherwise pull and adopt before the new
        # trainer has a chance to write its own outcome.
        rm -rf {bundle}
        mkdir -p {bundle_logs}
        if ! nvidia-smi -L >/dev/null 2>{bundle_logs}/nvidia-probe.log; then
          PROBE_TAIL=$(tr -d '\\n' < {bundle_logs}/nvidia-probe.log)
          echo "odin: gpu_unavailable: $PROBE_TAIL" > {bundle_logs}/odin-submit-error.log
          echo "odin: gpu_unavailable: $PROBE_TAIL" >&2
          exit 1
        fi
        nohup setsid bash -c '{inner_inner}' > /dev/null 2>&1 < /dev/null &
        TRAINING_PID=$!
        cat > {bundle}/.tracker.json <<TRACKER_EOF
        {{
          "schema_version": "{TRACKER_SCHEMA_VERSION}",
          "run_id": "{job.run_id}",
          "container_name": "{host.container_name}",
          "host": "{host.host}",
          "submitted_at": "{submitted_at}",
          "pid": $TRAINING_PID,
          "container_pid": null,
          "per_job_timeout_s": {per_job_timeout_s}
        }}
        TRACKER_EOF
        echo "odin-submit: ok run_id={job.run_id} bundle={job.bundle_dir_name}"
        """
    )
    return (
        f"cd {host.isaaclab_path} && "
        f"docker exec -i {host.container_name} bash -l <<'ASGARD_SUBMIT_EOF'\n"
        f"{body}"
        f"ASGARD_SUBMIT_EOF"
    )


def _build_poll_script(host: ValkyrieConfig, bundle_ids: list[str]) -> str:
    """Return one batched poll command for all in-flight bundles on ``host``.

    The command iterates each bundle and emits one of four states:

      - ``done``: ``manifest.json`` exists (training wrote its bundle).
      - ``alive``: no manifest, ``.run.pid`` exists, and ``kill -0`` finds
        the PID — training still running.
      - ``exited-no-manifest``: no manifest, ``.run.pid`` exists, but the
        PID is gone — training crashed (or was killed by sweep_timeouts).
      - ``no-pidfile``: neither manifest nor pidfile — sub-second window
        between submit and the inner-inner shell writing ``.run.pid``;
        keep polling.

    A note on safety: the body uses no single quotes (so the outer
    ``bash -lc '...'`` works) and only standard POSIX file tests +
    ``kill -0``. No PTY required.

    Args:
        host: Valkyrie whose bundles we're polling.
        bundle_ids: ``bundle_dir_name`` of every in-flight job on this host.

    Returns:
        SSH command to pass to :meth:`SSHRunner.run` with ``pty=False``.
    """
    if not bundle_ids:
        raise ValueError("_build_poll_script requires at least one bundle_id")
    bundles = " ".join(bundle_ids)
    inner = (
        f"for bundle in {bundles}; do "
        f"if [ -f /workspace/isaaclab/odin_runs/$bundle/manifest.json ]; then "
        f'echo "$bundle {POLL_DONE}"; '
        f"elif [ -f /workspace/isaaclab/odin_runs/$bundle/.run.pid ]; then "
        f"pid=$(cat /workspace/isaaclab/odin_runs/$bundle/.run.pid); "
        f'if kill -0 "$pid" 2>/dev/null; then '
        f'echo "$bundle {POLL_ALIVE}"; '
        f"else "
        f'echo "$bundle {POLL_EXITED_NO_MANIFEST}"; '
        f"fi; "
        f"else "
        f'echo "$bundle {POLL_NO_PIDFILE}"; '
        f"fi; "
        f"done"
    )
    return f"cd {host.isaaclab_path} && docker exec {host.container_name} bash -lc '{inner}'"


def _parse_poll_output(stdout: str) -> dict[str, str]:
    """Decode the per-line ``<bundle> <state>`` poll stdout into a dict.

    Garbage lines (SSH banner, empty lines, unknown states) are dropped.

    Args:
        stdout: Raw stdout from the poll SSH call.

    Returns:
        Mapping ``bundle_id → state`` containing only recognised
        :data:`POLL_*` strings.
    """
    states: dict[str, str] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        bundle, state = parts
        if state in _POLL_STATES:
            states[bundle] = state
    return states


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
        # Detached-mode in-memory record of submitted jobs awaiting terminal
        # poll outcome. Single-job-per-host policy keeps this dict at ≤1
        # entry today, but the structure supports >1 in case the policy
        # changes. Keyed by ``run_id``.
        self._inflight: dict[str, JobInflight] = {}
        # Kill requests pushed by the runner via ``request_cancel(run_id)``.
        # Drained on each ``_sweep_cancellations`` tick.
        self._cancel_request: dict[str, bool] = {}
        self._cancel_request_lock = threading.Lock()
        # Set transiently by ``_try_take_job`` when the sentinel is consumed
        # so the run loop can flip its own sentinel-seen flag without
        # juggling tuple returns.
        self._sentinel_just_seen: bool = False

    # -- public entry point -------------------------------------------------

    def run(self) -> None:
        if self._options.detached_mode:
            self._run_detached()
        else:
            self._run_legacy_pty()

    def _run_legacy_pty(self) -> None:
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

    def _run_detached(self) -> None:
        """Detached-mode run loop: submit-and-poll.

        Single-job-per-host policy keeps :attr:`_inflight` at ≤1 entry.
        The loop alternates: pull a job from the queue if we have
        capacity → submit → poll → finalize. Sentinel handling is
        slightly trickier than the legacy path: receiving ``None`` does
        NOT exit immediately because we may still have an in-flight job
        whose terminal poll is pending. Exit only when both the sentinel
        is in hand AND :attr:`_inflight` is empty.
        """
        sentinel_seen = False
        while not self._shutdown.is_set() and not self._down_event.is_set():
            if not sentinel_seen and not self._inflight:
                job = self._try_take_job()
                if job is None and self._sentinel_just_seen:
                    sentinel_seen = True
                elif job is not None:
                    self._submit_or_handle(job)
            elif self._inflight:
                self._sweep_cancellations()
                self._poll_inflight_once()
                self._sweep_timeouts()
            if sentinel_seen and not self._inflight:
                return
            if self._inflight:
                # Wait one poll-interval between ticks; short-sleep
                # tightens the loop in tests (poll_interval_s=0).
                time.sleep(self._options.poll_interval_s)
            else:
                # No work in flight, queue empty: short sleep so we
                # don't busy-spin while waiting for the sentinel.
                time.sleep(0.1 if self._options.poll_interval_s else 0)

    def _try_take_job(self) -> JobEntry | None:
        """Pop one job from the queue (non-blocking).

        Sets :attr:`_sentinel_just_seen` when the sentinel arrives so the
        caller can flip its sentinel-seen flag. Handles the
        ``preferred_not`` re-queue case in-line.

        Returns:
            The popped job, or ``None`` when the queue is empty / a
            sentinel was just consumed / the job was put back.
        """
        self._sentinel_just_seen = False
        try:
            job = self._job_queue.get_nowait()
        except queue.Empty:
            return None
        if job is None:
            self._sentinel_just_seen = True
            return None
        if job.preferred_not and self.host.host in job.preferred_not:
            seen_count = self._preferred_not_seen.get(job.run_id, 0) + 1
            self._preferred_not_seen[job.run_id] = seen_count
            if seen_count < 3:
                self._job_queue.put(job)
                return None
            # Fall through: bounded fallback exhausted, take it.
        return job

    def _submit_or_handle(self, job: JobEntry) -> None:
        """Submit one job. On success, register inflight; on terminal failure,
        emit the matching :class:`StateEvent` (with the existing
        recovery / retry / quarantine policy applied)."""
        # Skip race: between when this job was put on the queue and when we
        # popped it off, the runner may have flipped its status to 'failed'
        # in response to an operator skip. Re-check before paying for an
        # SSH submit.
        if job.status != "pending":
            return
        started_at = _utc_now_iso()
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="running",
                started_at=started_at,
            )
        )
        job.started_at = started_at
        job.assigned_to = self.host.host
        job.attempts += 1
        result = self._submit_job(job)
        if result.ok:
            self._inflight[job.run_id] = JobInflight(
                job=job,
                tracker=None,  # populated lazily on first successful poll if needed
                submitted_at_monotonic=time.monotonic(),
            )
            return
        # Submit failed synchronously. Apply the same kind-driven policy
        # as the legacy path so retries / recovery / quarantine behave
        # identically.
        failure = result.failure
        if failure is None:
            return
        self._handle_synchronous_failure(job, failure)

    def _handle_synchronous_failure(self, job: JobEntry, failure: FailureInfo) -> None:
        """Apply retry / recovery / quarantine to a synchronous submit failure.

        Mirrors the kind-driven branches in :meth:`_execute` so the
        operator-visible behaviour is the same on both paths.
        """
        if failure.kind == "infrastructure":
            if job.attempts <= self._options.max_infrastructure_retries:
                # Re-queue the job for another attempt on this same host.
                # (Mirrors the legacy retry loop.)
                self._job_queue.put(job)
                return
        elif failure.kind == "gpu_lost":
            rec = recover_valkyrie_gpu(self.host, ssh=self._ssh)
            if rec.recovered:
                self._state_chan.put(StateEvent(run_id=job.run_id, host=self.host.host, transition="recovered"))
                if job.attempts <= self._options.max_infrastructure_retries:
                    self._job_queue.put(job)
                    return
            else:
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
                return
        # Quarantine bookkeeping (host-health failures) before terminal-fail.
        ssh_tail = self._dispatch_dir / job.bundle_dir_name / "logs" / "ssh-tail.log"
        if self._quarantine_check_and_handle(job=job, failure=failure, ssh_tail=ssh_tail):
            return
        self._emit_failed(job, failure)

    def _poll_inflight_once(self) -> None:
        """One poll tick: query the host once for all inflight bundles, then
        finalize any that landed on a terminal state."""
        bundle_to_run: dict[str, str] = {
            inflight.job.bundle_dir_name: run_id for run_id, inflight in self._inflight.items()
        }
        if not bundle_to_run:
            return
        states = self._poll_host(list(bundle_to_run.keys()))
        for bundle_id, state in states.items():
            run_id = bundle_to_run.get(bundle_id)
            if run_id is None:
                continue
            inflight = self._inflight.get(run_id)
            if inflight is None:
                continue
            if state in (POLL_DONE, POLL_EXITED_NO_MANIFEST):
                self._finalize_terminal(inflight, state)

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
            ssh_result = self._ssh.run(self.host, cmd, timeout_s=float(self._timeout_for(job)), stdout_tee=ssh_tail)

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
            if job.transition_to("failed", failure=failure):
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
            if job.transition_to("failed", failure=rsync_failure):
                self._state_chan.put(
                    StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=rsync_failure)
                )
            return

        # Validate the bundle: manifest.json present, schema-v1 shape.
        bundle_failure = _validate_bundle(local_bundle)
        if bundle_failure is not None:
            if self._quarantine_check_and_handle(job=job, failure=bundle_failure, ssh_tail=ssh_tail):
                return
            if job.transition_to("failed", failure=bundle_failure):
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

    # -- per-job budget -----------------------------------------------------

    def _timeout_for(self, job: JobEntry) -> int:
        """Resolve the wall-clock timeout for ``job``.

        Per-job override (populated by the runner from
        :func:`tools.odin.asgard.budgets.Budgets.lookup`) wins. Falls back
        to the dispatcher-wide default in :attr:`WorkerOptions.per_job_timeout_s`
        so existing single-timeout dispatches keep working unchanged.
        """
        if job.per_job_timeout_s is not None:
            return job.per_job_timeout_s
        return self._options.per_job_timeout_s

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

    # -- detached: submit ---------------------------------------------------

    def _submit_job(self, job: JobEntry) -> SubmitResult:
        """Run the detached-submit phase for one job.

        Sends the script from :func:`_build_submit_script` over SSH with
        ``pty=False`` and short retries on transient SSH-connect failures.
        Recognised return modes:

          - SSH exit 0 + stdout contains the ``odin-submit: ok`` sentinel
            → trainer is detached on the remote; tracker is in place.
          - SSH exit 1 + stderr contains ``odin: gpu_unavailable`` → GPU
            probe failed pre-flight; classify ``gpu_lost`` for the
            recovery path to fire.
          - SSH exit 125/126/127 → docker daemon couldn't dispatch;
            classify ``infrastructure``.
          - Transient SSH failure (exit 255 = ``ssh: connect``) → retry up
            to ``submit_max_retries`` with a short sleep between attempts.
          - All other non-zero exits → classify ``hugin_crash`` (the
            script body itself failed before backgrounding).

        Args:
            job: Job to submit.

        Returns:
            :class:`SubmitResult`; on success the worker should record an
            inflight entry and start polling; on failure the worker should
            terminal-fail the job (or trip its retry/quarantine logic).
        """
        script = _build_submit_script(
            self.host,
            job,
            submitted_at=_utc_now_iso(),
            per_job_timeout_s=self._timeout_for(job),
        )
        last_failure: FailureInfo | None = None
        for attempt in range(1, self._options.submit_max_retries + 1):
            r = self._ssh.run(self.host, script, timeout_s=120.0, pty=False)
            if r.exit_code == 0 and _SUBMIT_OK_PREFIX in r.stdout:
                return SubmitResult(ok=True)
            if r.exit_code == 1 and any(sig in r.stderr for sig in _GPU_LOST_SIGNATURES):
                return SubmitResult(
                    ok=False,
                    failure=FailureInfo(
                        kind="gpu_lost",
                        message="GPU probe failed at submit",
                        details={"attempts": attempt, "stderr_tail": r.stderr.strip()[-200:]},
                    ),
                )
            if r.exit_code in _INFRASTRUCTURE_DOCKER_EXIT_CODES:
                return SubmitResult(
                    ok=False,
                    failure=FailureInfo(
                        kind="infrastructure",
                        message=f"docker exec failed with exit {r.exit_code}: {r.stderr.strip() or 'unknown'}",
                        details={"exit_code": r.exit_code, "attempts": attempt},
                    ),
                )
            if r.exit_code == 255:
                # Transient ssh-side glitch (connection refused / timed
                # out). Retry up to the cap before giving up.
                last_failure = FailureInfo(
                    kind="infrastructure",
                    message=f"ssh transient error: {r.stderr.strip() or 'exit 255'}",
                    details={"exit_code": r.exit_code, "attempts": attempt},
                )
                if attempt < self._options.submit_max_retries:
                    time.sleep(self._options.submit_retry_backoff_s)
                continue
            return SubmitResult(
                ok=False,
                failure=FailureInfo(
                    kind="hugin_crash",
                    message=(
                        f"submit script exited {r.exit_code}; stderr tail:"
                        f" {repr(r.stderr.strip().splitlines()[-1]) if r.stderr.strip() else '(empty)'}"
                    ),
                    details={"exit_code": r.exit_code, "attempts": attempt},
                ),
            )
        return SubmitResult(ok=False, failure=last_failure)

    # -- detached: poll -----------------------------------------------------

    def _poll_host(self, bundle_ids: list[str]) -> dict[str, str]:
        """One short SSH per host per tick. Returns ``bundle → poll-state``.

        Failures (network blips) return an empty dict so the run loop
        treats it as "no terminal news this tick" and the next tick will
        catch up.

        Args:
            bundle_ids: ``bundle_dir_name``s to query.

        Returns:
            Mapping limited to the four recognised :data:`POLL_*` states.
        """
        if not bundle_ids:
            return {}
        cmd = _build_poll_script(self.host, bundle_ids)
        r = self._ssh.run(self.host, cmd, timeout_s=30.0, pty=False)
        if r.exit_code != 0:
            return {}
        return _parse_poll_output(r.stdout)

    def _read_local_log(self, job: JobEntry, filename: str) -> str:
        """Read a log file from the locally-pulled bundle, returning ``""`` on miss.

        Used by :meth:`_classify_remote` after the bundle was rsynced
        back. Reading locally avoids a second SSH round trip on the
        terminal-failure path; the rsync pull is already on the critical
        path, so the file is already present (or genuinely absent because
        the trainer never wrote it).
        """
        path = self._dispatch_dir / job.bundle_dir_name / "logs" / filename
        if not path.exists():
            return ""
        try:
            return path.read_text()
        except OSError:
            return ""

    def _classify_remote_text(self, text: str) -> FailureInfo:
        """Instance-method shim around :func:`classify_remote_stderr`.

        Kept on the worker so tests can call it directly without resolving
        the bundle dir. Implementation is delegated to the free function
        so reconcile (which has no worker instance) can apply the same
        rules.
        """
        return classify_remote_stderr(text)

    def _classify_remote(self, job: JobEntry) -> FailureInfo:
        """Read the just-pulled bundle's stderr files and classify the failure.

        Concatenates ``<bundle>/logs/odin-submit-error.log`` and
        ``<bundle>/logs/hugin-stderr.log`` (whichever the trainer
        produced), then runs the combined text through
        :meth:`_classify_remote_text`.

        Reads from the local rsync target rather than ``docker exec cat``
        — the bundle was already pulled on the terminal-failure path, so
        the files are already on disk.

        Args:
            job: The failing job (used to locate the bundle dir).

        Returns:
            ``FailureInfo`` describing the failure.
        """
        submit_err = self._read_local_log(job, "odin-submit-error.log")
        train_err = self._read_local_log(job, "hugin-stderr.log")
        return self._classify_remote_text(f"{submit_err}\n{train_err}")

    # -- detached: finalize / sweep -----------------------------------------

    def _finalize_terminal(self, inflight: JobInflight, poll_state: str) -> None:
        """Drive a job through to a terminal :class:`StateEvent`.

        Two terminal poll states feed into this method:

          - :data:`POLL_DONE`: rsync-pull the bundle, validate the
            manifest, emit ``completed`` (or a malformed-bundle failure).
          - :data:`POLL_EXITED_NO_MANIFEST`: rsync-pull the bundle (best
            effort), classify via :meth:`_classify_remote`, emit
            ``failed``. Classification precedence (highest first):
            ``inflight.timeout_kill_dispatched`` → ``kind="timeout"``;
            ``inflight.kill_dispatched`` (operator kill via
            :meth:`request_cancel`) → ``kind="killed"``; otherwise →
            :meth:`_classify_remote`.

        Removes the entry from :attr:`_inflight` on the way out.
        """
        job = inflight.job
        remote_bundle = f"{self.host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        local_bundle = self._dispatch_dir / job.bundle_dir_name
        rsync_result = self._rsync.pull(self.host, remote_bundle, local_bundle)

        if poll_state == POLL_DONE:
            if rsync_result.exit_code != 0:
                failure = FailureInfo(
                    kind="infrastructure",
                    message=f"rsync pull failed: {rsync_result.stderr.strip() or 'non-zero exit'}",
                    details={"attempts": job.attempts},
                )
                self._emit_failed(job, failure)
                self._inflight.pop(job.run_id, None)
                return
            bundle_failure = _validate_bundle(local_bundle)
            if bundle_failure is not None:
                self._emit_failed(job, bundle_failure)
                self._inflight.pop(job.run_id, None)
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
            self._inflight.pop(job.run_id, None)
            return

        # POLL_EXITED_NO_MANIFEST
        if inflight.timeout_kill_dispatched:
            # Timeout precedence stays — operator-clicked Kill on a job that
            # tripped its budget gets the more accurate kind="timeout".
            timeout_s = self._timeout_for(job)
            failure = FailureInfo(
                kind="timeout",
                message=f"remote process exceeded {timeout_s}s",
                details={"per_job_timeout_s": timeout_s},
            )
        elif inflight.kill_dispatched:
            failure = FailureInfo(
                kind="killed",
                message="operator kill",
                details={"per_job_timeout_s": self._options.per_job_timeout_s},
            )
        else:
            failure = self._classify_remote(job)
        self._emit_failed(job, failure)
        self._inflight.pop(job.run_id, None)

    def _emit_failed(self, job: JobEntry, failure: FailureInfo) -> None:
        """Stamp the job as ``failed`` and post the matching :class:`StateEvent`.

        Self-loops (already-failed jobs) short-circuit at the helper level
        and post no event — the worker should not be calling _emit_failed
        on a job that's already terminal, but the no-op safety net is
        cheap insurance.
        """
        if not job.transition_to("failed", failure=failure):
            return
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="failed",
                failure=failure,
                ended_at=job.ended_at,
            )
        )

    def request_cancel(self, run_id: str) -> None:
        """Mark ``run_id`` for kill. Called by the runner from its main thread.

        Thread-safe: a single dict assignment is atomic in CPython, but the
        explicit lock keeps the contract obvious and protects against
        concurrent ``_sweep_cancellations`` reads during list-rebuild.
        """
        with self._cancel_request_lock:
            self._cancel_request[run_id] = True

    def _sweep_cancellations(self) -> None:
        """For each pending kill request, dispatch a best-effort pkill once.

        The next poll tick will see ``exited-no-manifest`` and
        :meth:`_finalize_terminal` will classify as ``killed`` (because
        ``inflight.kill_dispatched`` is set here).
        """
        with self._cancel_request_lock:
            requested = list(self._cancel_request.keys())
            self._cancel_request.clear()
        for run_id in requested:
            inflight = self._inflight.get(run_id)
            if inflight is None:
                # Job already finished (or was never on this worker). Drop.
                continue
            if inflight.kill_dispatched:
                continue
            self._cleanup_remote_process(inflight.job)
            inflight.kill_dispatched = True

    def _sweep_timeouts(self) -> None:
        """For each in-flight job past its budget, dispatch a best-effort kill.

        We don't terminal-fail here — the next poll tick will see
        ``exited-no-manifest`` and :meth:`_finalize_terminal` will use the
        ``timeout_kill_dispatched`` flag to classify as ``timeout``.
        """
        now = time.monotonic()
        for inflight in list(self._inflight.values()):
            if inflight.timeout_kill_dispatched:
                continue
            elapsed = now - inflight.submitted_at_monotonic
            if elapsed < self._timeout_for(inflight.job):
                continue
            self._cleanup_remote_process(inflight.job)
            inflight.timeout_kill_dispatched = True

    # -- classification -----------------------------------------------------

    def _classify(self, r: SSHResult, job: JobEntry, ssh_tail: Path) -> FailureInfo | None:
        if r.timed_out:
            timeout_s = self._timeout_for(job)
            return FailureInfo(
                kind="timeout",
                message=f"remote process exceeded {timeout_s}s",
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
    """Validate bundle structure AND training-phase outcome.

    Schema-level checks (``hugin_malformed_bundle``) catch transport /
    framework breakage. The training-phase content check
    (``hugin_crash``) catches a subtler failure mode observed in
    production: a SIGKILL'd orphan trainer can leave behind a
    schema-valid manifest whose ``phases.training.status=failed,
    exit_code=-9``. Without inspecting those two fields the worker
    would adopt the failed run as completed on the next dispatch's
    first poll.
    """
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
    training = m.get("phases", {}).get("training", {})
    train_status = training.get("status")
    train_exit = training.get("exit_code")
    if train_status != "completed" or (train_exit is not None and train_exit != 0):
        return FailureInfo(
            kind="hugin_crash",
            message=(
                f"manifest reports training phase did not complete cleanly: "
                f"status={train_status!r} exit_code={train_exit!r}"
            ),
            details={
                "bundle_dir": str(local_bundle.name),
                "exit_code": train_exit,
                "training_status": train_status,
            },
        )
    return None
