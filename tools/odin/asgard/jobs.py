# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queue construction — expand curated env lists across seeds into JobEntry rows."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from tools.odin.common.env_list import load_env_list

__all__ = ["JobEntry", "FailureInfo", "SkippedEntry", "build_queue_from_env_lists"]


def _utc_now_iso() -> str:
    """UTC timestamp in ``YYYY-MM-DDTHH:MM:SSZ`` form. Mirrors the
    runner's ``_utc_now_iso`` so both modules produce identical strings."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FailureInfo:
    """Classified failure attached to a :class:`JobEntry` when ``status == 'failed'``.

    ``kind`` values:

    - ``infrastructure``: docker / SSH transport failure (retried).
    - ``hugin_crash``: training process exited non-zero with no
      Odin-recognised stderr signal. Also covers Hugin's silent-exit-0
      case (returncode 0 but no output JSON), which ``_run_phase``
      promotes to a non-zero exit before ``main()`` returns.
    - ``hugin_malformed_bundle``: SSH succeeded, rsync pulled, but the
      bundle's manifest is missing or invalid.
    - ``timeout``: SSH wall-clock timeout fired.
    - ``preset_unsupported``: training process exited non-zero with a
      stderr line beginning ``preset_unsupported:`` — the requested
      preset doesn't exist for the task. Caught by the runtime safety
      net when yaml-stamped ``presets_available`` is stale.
    - ``gpu_lost``: training process exited non-zero with a GPU-loss
      signature in stderr (NVML init failure, CUDA "no device", Vulkan
      driver mismatch). Worker attempts container-restart-based
      recovery before retrying on the same host. Counts against
      ``max_infrastructure_retries``.
    - ``killed``: operator-initiated kill via the Tab A cancel button.
      The worker pkilled the trainer mid-run and pulled whatever
      partial bundle was on disk. Does NOT count as a host-health
      failure for the circuit-breaker.
    - ``skipped``: operator-initiated skip via the Tab A cancel button.
      The runner flipped a pending job to failed before any worker
      submitted. No bundle exists.
    """

    kind: str
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class SkippedEntry:
    """An (task, framework, backend, seed) pair that the queue builder rejected.

    Lives next to :class:`JobEntry` because both are persisted into
    ``dispatch.json`` (jobs[] and skipped[] respectively).  ``reason``
    values today: ``"preset_unsupported"`` (yaml's
    ``presets_available`` excludes the requested backend) and
    ``"native_backend_mismatch"`` (no preset system, native backend
    doesn't match request).  Optional ``native_backend`` carries
    additional telemetry when ``reason="native_backend_mismatch"``.
    """

    task_id: str
    framework: str
    backend: str
    seed: int
    reason: str
    presets_available: list[str] = field(default_factory=list)
    native_backend: str | None = None


@dataclass
class JobEntry:
    """One row in the dispatch queue — the smallest unit of work."""

    run_id: str
    task_id: str
    framework: str  # "rsl_rl" | "skrl"
    backend: str  # "physx" | "newton"
    num_envs: int
    max_iterations: int
    seed: int
    bundle_dir_name: str
    status: str = "pending"  # pending | assigned | running | completed | failed
    assigned_to: str | None = None
    attempts: int = 0
    failure: FailureInfo | None = None
    preferred_not: set[str] = field(default_factory=set)
    started_at: str | None = None
    ended_at: str | None = None
    # Substate annotation while status == "running". Distinguishes
    # the active-training phase from finalization (rsync.pull). Renderers
    # use this to show a "pulling bundle" badge without changing status.
    running_substate: str | None = None  # "training" | "pulling_bundle" | None
    # Per-job wall-clock timeout in seconds. ``None`` means "use the
    # dispatcher-wide :attr:`DispatchOptions.per_job_timeout_s`". Populated by
    # :func:`tools.odin.asgard.runner.run_dispatch` from
    # :func:`tools.odin.asgard.budgets.Budgets.lookup` so heavy tasks (Allegro,
    # Camera) get hours instead of the global 1h default while Cartpole stays
    # at 10 minutes.
    per_job_timeout_s: int | None = None
    # OSMO task name for jobs submitted via the Bifrost dispatcher. ``None``
    # for all Asgard-dispatched jobs.
    osmo_task_name: str | None = None
    # Last heartbeat timestamp emitted by the worker thread for this job.
    # Set by ``_apply_state_event`` on ``StateEvent(transition='heartbeat')``.
    # Heimdall reads this to detect stale jobs whose worker thread is wedged
    # (rsync hang, blocking SSH call, etc.). ``None`` on pre-Heimdall
    # dispatches and on jobs that have not yet emitted any heartbeat.
    last_heartbeat_at: str | None = None

    # Allowed-transition graph. See spec §4.1. Self-loops are not listed
    # here — `transition_to` short-circuits same-state calls as no-ops
    # before consulting this map.
    _ALLOWED_TRANSITIONS: ClassVar[dict[str, frozenset[str]]] = {
        # back-compat: legacy dispatch.json may carry "assigned"; reset_in_flight_to_pending flips it to pending
        "assigned": frozenset({"pending"}),
        "completed": frozenset({"pending"}),
        "failed": frozenset({"pending"}),
        # ``pending → completed`` and ``pending → failed`` are the OSMO/Bifrost
        # fast-path: a workflow scheduler can report a task as terminal
        # without bifrost ever observing it in the "running" state (cache
        # hits, fast scheduling, instant failures). The Asgard worker path
        # always goes through running first, so this edge is OSMO-only in
        # practice.
        "pending": frozenset({"running", "completed", "failed"}),
        "running": frozenset({"completed", "failed", "pending"}),
    }

    def transition_to(
        self,
        target: str,
        *,
        failure: FailureInfo | None = None,
        assigned_to: str | None = None,
        now: str | None = None,
        reset_attempts: bool = False,
        add_preferred_not: str | None = None,
    ) -> bool:
        """Transition this job to ``target`` per spec §4.2.

        Validates the (current, target) edge against
        :data:`_ALLOWED_TRANSITIONS`. Self-loops short-circuit as
        no-ops and return ``False``. Legal cross-state edges apply
        the per-target field contract and return ``True``. Illegal
        edges raise ``ValueError``.

        Args:
            target: One of ``"pending"`` | ``"running"`` | ``"completed"`` | ``"failed"``.
            failure: Required when ``target == "failed"``. Forbidden
                when ``target == "completed"``.
            assigned_to: Required when ``target == "running"``.
            now: ISO-8601 UTC timestamp; defaults to :func:`_utc_now_iso`.
            reset_attempts: Only honored when ``target == "pending"``.
                When True, zeros ``attempts``.
            add_preferred_not: Only honored when ``target == "pending"``.
                When set, adds the host to ``preferred_not``.

        Returns:
            ``True`` when a cross-state edge applied (fields mutated).
            ``False`` for self-loops (no mutation).

        Raises:
            ValueError: For illegal edges or contract violations.
        """
        # Self-loop short-circuit.
        if target == self.status:
            return False

        # Legality check.
        allowed = self._ALLOWED_TRANSITIONS.get(self.status, frozenset())
        if target not in allowed:
            raise ValueError(
                f"illegal transition {self.status!r} → {target!r} for run_id={self.run_id!r}; "
                f"allowed targets from {self.status!r}: {sorted(allowed)}"
            )

        # Contract checks.
        if target == "running" and assigned_to is None:
            raise ValueError(f"transition_to('running') requires assigned_to (run_id={self.run_id!r})")
        if target == "failed" and failure is None:
            raise ValueError(f"transition_to('failed') requires failure (run_id={self.run_id!r})")
        if target == "completed" and failure is not None:
            raise ValueError(f"transition_to('completed') must not pass failure (run_id={self.run_id!r})")

        ts = now if now is not None else _utc_now_iso()

        # Apply per-target field contract.
        if target == "pending":
            self.status = "pending"
            self.started_at = None
            self.ended_at = None
            self.assigned_to = None
            self.failure = None
            self.running_substate = None
            if reset_attempts:
                self.attempts = 0
            if add_preferred_not is not None:
                # `preferred_not` is a set; copy-on-write to avoid mutating
                # any caller's reference accidentally shared.
                self.preferred_not = set(self.preferred_not) | {add_preferred_not}
        elif target == "running":
            self.status = "running"
            self.started_at = ts
            self.assigned_to = assigned_to
            self.ended_at = None
            self.failure = None
            self.running_substate = "training"
        elif target == "completed":
            self.status = "completed"
            self.ended_at = ts
            self.failure = None
            self.running_substate = None
        elif target == "failed":
            self.status = "failed"
            self.ended_at = ts
            self.failure = failure
            self.running_substate = None

        return True


def _framework_slug(framework: str) -> str:
    """rsl_rl -> rsl-rl, skrl -> skrl (hyphen variant used in run_id paths)."""
    return framework.replace("_", "-")


def _make_run_id(framework: str, backend: str, task_id: str, dispatch_id: str, seed: int) -> str:
    return f"{_framework_slug(framework)}_{backend}_{task_id}_{dispatch_id}_seed{seed}"


def _apply_include_filter(task_id: str, include_filter: list[str] | None) -> bool:
    if not include_filter:
        return True
    return any(fnmatch.fnmatch(task_id, pat) for pat in include_filter)


def _expand_env_list(
    yaml_path: Path,
    backend: str,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None,
) -> tuple[list[JobEntry], list[SkippedEntry]]:
    env_list = load_env_list(yaml_path)
    jobs: list[JobEntry] = []
    skipped: list[SkippedEntry] = []
    for group_rows in env_list.groups.values():
        for row in group_rows:
            if not row.keep or row.status == "stale":
                continue
            if not _apply_include_filter(row.task_id, include_filter):
                continue
            if row.framework is None or row.num_envs is None or row.max_iterations is None:
                continue
            # Rule 1: preset system says backend not supported.
            if row.presets_available and backend not in row.presets_available:
                for seed in seeds:
                    skipped.append(
                        SkippedEntry(
                            task_id=row.task_id,
                            framework=row.framework,
                            backend=backend,
                            seed=seed,
                            reason="preset_unsupported",
                            presets_available=list(row.presets_available),
                            native_backend=row.native_backend,
                        )
                    )
                continue

            # Rule 2: no preset system, native_backend known and mismatching
            # the requested backend → silent-swap prevention.
            if not row.presets_available and row.native_backend is not None and row.native_backend != backend:
                for seed in seeds:
                    skipped.append(
                        SkippedEntry(
                            task_id=row.task_id,
                            framework=row.framework,
                            backend=backend,
                            seed=seed,
                            reason="native_backend_mismatch",
                            presets_available=[],
                            native_backend=row.native_backend,
                        )
                    )
                continue

            for seed in seeds:
                run_id = _make_run_id(row.framework, backend, row.task_id, dispatch_id, seed)
                jobs.append(
                    JobEntry(
                        run_id=run_id,
                        task_id=row.task_id,
                        framework=row.framework,
                        backend=backend,
                        num_envs=row.num_envs,
                        max_iterations=row.max_iterations,
                        seed=seed,
                        bundle_dir_name=run_id,
                    )
                )
    return jobs, skipped


def build_queue_from_env_lists(
    physx_yaml: Path | None,
    newton_yaml: Path | None,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None = None,
) -> tuple[list[JobEntry], list[SkippedEntry]]:
    """Expand curated env YAMLs across seeds into a flat ``(jobs, skipped)`` pair.

    Args:
        physx_yaml: Path to ``physx_envs.yaml`` (T2.1); ``None`` to skip PhysX.
        newton_yaml: Path to ``newton_envs.yaml`` (T2.1); ``None`` to skip Newton.
        seeds: Seeds to expand each kept row across. Must be non-empty.
        dispatch_id: UTC timestamp (``YYYYMMDD-HHMMSS``) shared by all run_ids
            in this dispatch.
        include_filter: Optional list of fnmatch patterns on ``task_id``; a row
            must match at least one pattern to be queued. Unset = keep all.

    Returns:
        ``(jobs, skipped)``. ``jobs`` is a list of :class:`JobEntry` rows in
        insertion order (PhysX first, then Newton). ``skipped`` is the list
        of :class:`SkippedEntry` rows for ``(task, backend, seed)`` triples
        rejected by the queue filter — either because the row's
        ``presets_available`` excludes the requested backend
        (``reason="preset_unsupported"``) or because the row has no
        preset system and its ``native_backend`` doesn't match the request
        (``reason="native_backend_mismatch"``). Each ``--include``-passing
        seed contributes one ``SkippedEntry``.

    Raises:
        ValueError: If neither YAML is provided or seeds is empty.
    """
    if physx_yaml is None and newton_yaml is None:
        raise ValueError("build_queue_from_env_lists needs at least one env list (physx_yaml or newton_yaml)")
    if not seeds:
        raise ValueError("build_queue_from_env_lists needs a non-empty seed list")

    jobs: list[JobEntry] = []
    skipped: list[SkippedEntry] = []
    if physx_yaml is not None:
        j, s = _expand_env_list(physx_yaml, "physx", seeds, dispatch_id, include_filter)
        jobs.extend(j)
        skipped.extend(s)
    if newton_yaml is not None:
        j, s = _expand_env_list(newton_yaml, "newton", seeds, dispatch_id, include_filter)
        jobs.extend(j)
        skipped.extend(s)
    return jobs, skipped
